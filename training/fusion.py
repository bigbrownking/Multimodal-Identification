"""
Слияние отпечатка и радужки на мультимодальных корпусах виртуальных субъектов.

  python training/fusion.py --corpus multimodal_sd302    --fp-run runs/sd302_r50_reg    --iris-run runs/iris_resnet50
  python training/fusion.py --corpus multimodal_socofing --fp-run runs/socofing_resnet50 --iris-run runs/iris_resnet50

Берутся уже обученные мономодальные модели (best.pt), их веса не меняются.
Сравниваются на одних и тех же парах «тот же субъект / разные субъекты»:
  fp_only        — только отпечаток (косинус эмбеддингов)
  iris_only      — только радужка
  score_sum      — слияние оценок: z-нормализация (параметры по val) + сумма с весом 0,5
  score_weighted — то же с весом w, подобранным по минимуму EER на val
  feat_concat    — слияние признаков без обучения: косинус склеенных L2-векторов (= среднее косинусов)
  feat_learned   — слияние признаков с обучением: [e_fp, e_iris] (1024) -> Dropout -> Linear 512 -> BN,
                   голова ArcFace по виртуальным субъектам train; лучшая эпоха по EER на val
Порог для FAR/FRR/accuracy/F1 на test — EER-порог каждого метода на val.
Результаты: runs/fusion_<корпус>/ (summary.md, metrics.json, графики) + MLflow (эксперимент week8_fusion).
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.datasets import FP_SIZE, IRIS_SIZE, OUTPUT_SPLITS, read_csv, read_gray, resolve, to_tensor  # noqa: E402
from training.train import (ArcFace, EmbeddingNet, Tee, Tracker, _plt, canonical, git_info,  # noqa: E402
                            impostor_pairs, genuine_pairs, operating_point, plot_roc_det, seed_all,
                            verification)

SPLITS = ("train", "val", "test")
METHODS = ("fp_only", "iris_only", "score_sum", "score_weighted", "feat_concat", "feat_learned")
LABELS = {"fp_only": "только отпечаток", "iris_only": "только радужка",
          "score_sum": "оценки: сумма (w=0,5)", "score_weighted": "оценки: взвешенная (w по val)",
          "feat_concat": "признаки: склейка", "feat_learned": "признаки: обученная проекция"}


# ======================================================================
# Эмбеддинги из обученных моделей
# ======================================================================
class PathDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[str], size: tuple[int, int]):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return to_tensor(read_gray(resolve(self.paths[i]), self.size)), 0


def load_run(run: Path, device: torch.device):
    ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    cfg["modality"] = canonical(cfg["modality"])
    model = EmbeddingNet(cfg["backbone"], cfg["embedding_dim"], pretrained=False, dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def embed(run: Path, paths: list[str], size, device, cache: Path, batch: int, workers: int) -> dict[str, np.ndarray]:
    best = run / "best.pt"
    known: dict[str, np.ndarray] = {}
    if cache.exists() and cache.stat().st_mtime >= best.stat().st_mtime:
        z = np.load(cache, allow_pickle=False)
        known = dict(zip(z["path"].tolist(), z["emb"].astype(np.float32)))
    todo = sorted(set(paths) - set(known))
    if todo:
        model, _ = load_run(run, device)
        loader = torch.utils.data.DataLoader(PathDataset(todo, size), batch_size=batch, num_workers=workers,
                                             pin_memory=device.type == "cuda")
        out = []
        t0 = time.perf_counter()
        for x, _ in loader:
            out.append(F.normalize(model(x.to(device, non_blocking=True)).float(), dim=1).cpu().numpy())
        E = np.concatenate(out)
        known.update(zip(todo, E))
        print(f"    {run.name}: посчитано {len(todo)} эмбеддингов за {time.perf_counter() - t0:.0f} с")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, path=np.array(list(known)), emb=np.stack(list(known.values())).astype(np.float16))
        del model
    else:
        print(f"    {run.name}: все {len(set(paths))} эмбеддингов из кэша")
    return known


# ======================================================================
# Методы слияния: одна функция для массивов пар и для матриц проба×галерея
# ======================================================================
def combine(method: str, sfp, sir, sfu, z: dict, w: float):
    zf = (sfp - z["fp_mu"]) / z["fp_sd"]
    zi = (sir - z["ir_mu"]) / z["ir_sd"]
    if method == "fp_only":
        return sfp
    if method == "iris_only":
        return sir
    if method == "score_sum":
        return 0.5 * zf + 0.5 * zi
    if method == "score_weighted":
        return w * zf + (1 - w) * zi
    if method == "feat_concat":
        return 0.5 * (sfp + sir)
    if method == "feat_learned":
        return sfu
    raise ValueError(method)


def rowdot(E: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 65536) -> np.ndarray:
    out = np.empty(len(a), np.float32)
    for s in range(0, len(a), chunk):
        out[s:s + chunk] = np.einsum("ij,ij->i", E[a[s:s + chunk]], E[b[s:s + chunk]])
    return out


class Split:
    def __init__(self, rows, efp, eir, fp_col):
        self.rows = rows
        self.vid = np.array([r["virtual_id"] for r in rows])
        self.y = np.unique(self.vid, return_inverse=True)[1]
        self.Efp = np.stack([efp[r[fp_col]] for r in rows]).astype(np.float32)
        self.Eir = np.stack([eir[r["iris_image_path"]] for r in rows]).astype(np.float32)
        self.Efu = None

    def pair_scores(self, a, b):
        sfu = rowdot(self.Efu, a, b) if self.Efu is not None else None
        return rowdot(self.Efp, a, b), rowdot(self.Eir, a, b), sfu

    def matrix_scores(self, p, g):
        sfu = self.Efu[p] @ self.Efu[g].T if self.Efu is not None else None
        return self.Efp[p] @ self.Efp[g].T, self.Eir[p] @ self.Eir[g].T, sfu


def make_pairs(sp: Split, n_imp: int, seed: int):
    ga, gb = genuine_pairs(sp.y)
    ia, ib = impostor_pairs(sp.y, n_imp, np.random.default_rng(seed))
    return (ga, gb), (ia, ib)


def gallery_rows(sp: Split) -> dict[int, int]:
    order = sorted(range(len(sp.rows)), key=lambda i: (sp.rows[i].get("fp_subset", "") != "Real",
                                                       sp.rows[i].get("fp_sample", ""),
                                                       sp.rows[i].get("iris_sample", "")))
    gal = {}
    for i in order:
        gal.setdefault(int(sp.y[i]), i)
    return gal


def identification(sp: Split, method, z, w, probe_idx=None):
    gal = gallery_rows(sp)
    g_idx = np.array([gal[c] for c in sorted(gal)])
    g_cls = sp.y[g_idx]
    pos = {c: k for k, c in enumerate(g_cls)}
    gset = set(g_idx.tolist())
    probes = np.array([i for i in (range(len(sp.y)) if probe_idx is None else probe_idx)
                       if i not in gset and sp.y[i] in pos], dtype=int)
    if len(probes) < 5:
        return None
    S = combine(method, *sp.matrix_scores(probes, g_idx), z, w)
    own = np.array([pos[sp.y[i]] for i in probes])
    gen = S[np.arange(len(own)), own].copy()
    S[np.arange(len(own)), own] = -np.inf
    rank1 = float(np.mean(gen > S.max(1)))
    k = min(5, S.shape[1])
    kth = -np.partition(-S, k - 2, axis=1)[:, k - 2] if k >= 2 else np.full(len(own), -np.inf)
    rank5 = float(np.mean(gen > kth))
    rng = np.random.default_rng(0)
    m = min(200000, 300 * len(own))
    pi, gj = rng.integers(0, len(own), m), rng.integers(0, len(g_idx), m)
    keep = gj != own[pi]
    res, _ = verification(gen, S[pi[keep], gj[keep]])
    return {"rank1": rank1, "rank5": rank5, "eer": res["eer"], "n_probes": int(len(own))}


# ======================================================================
# Слияние признаков с обучением
# ======================================================================
class FusionHead(nn.Module):
    def __init__(self, d_in: int = 1024, d: int = 512, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p), nn.Linear(d_in, d, bias=False), nn.BatchNorm1d(d))

    def forward(self, efp, eir):
        return self.net(torch.cat([efp, eir], 1))


@torch.no_grad()
def fused_embeddings(head, sp: Split, device) -> np.ndarray:
    head.eval()
    e = head(torch.from_numpy(sp.Efp).to(device), torch.from_numpy(sp.Eir).to(device))
    return F.normalize(e.float(), dim=1).cpu().numpy()


def train_learned(tr: Split, va: Split, va_pairs, args, device, log) -> tuple[FusionHead, list[dict]]:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    head = FusionHead(p=args.dropout).to(device)
    arc = ArcFace(512, int(tr.y.max()) + 1, args.s, args.m).to(device)
    opt = torch.optim.AdamW(list(head.parameters()) + list(arc.parameters()), lr=args.lr, weight_decay=args.wd)
    steps = math.ceil(len(tr.y) / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / steps) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(t / (args.epochs * steps), 1)))))
    Efp, Eir = torch.from_numpy(tr.Efp).to(device), torch.from_numpy(tr.Eir).to(device)
    y = torch.from_numpy(tr.y).long().to(device)
    by_vid = defaultdict(list)
    for i, c in enumerate(tr.y):
        by_vid[int(c)].append(i)
    by_vid = {c: np.array(v) for c, v in by_vid.items()}
    (ga, gb), (ia, ib) = va_pairs
    best, best_state, hist, bad = float("inf"), None, [], 0
    for ep in range(args.epochs):
        head.train(); arc.train()
        partner = np.array([rng.choice(by_vid[int(c)]) for c in tr.y])
        perm = torch.from_numpy(rng.permutation(len(tr.y))).to(device)
        pt = torch.from_numpy(partner).to(device)
        margin = args.m * min(1.0, ep / 3)
        tot, n = 0.0, 0
        for s in range(0, len(perm), args.batch):
            idx = perm[s:s + args.batch]
            if len(idx) < 2:
                continue
            logits, _ = arc(head(Efp[idx], Eir[pt[idx]]), y[idx], margin)
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        va.Efu = fused_embeddings(head, va, device)
        res, _ = verification(rowdot(va.Efu, ga, gb), rowdot(va.Efu, ia, ib))
        hist.append({"epoch": ep + 1, "loss": tot / max(n, 1), "val_eer": res["eer"]})
        better = res["eer"] < best
        if better:
            best, bad = res["eer"], 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or better:
            log(f"    эпоха {ep + 1:3d}: loss {tot / max(n, 1):.3f}, val EER {100 * res['eer']:.2f}%"
                + ("  <- лучшая" if better else ""))
        if bad >= args.patience:
            log(f"    ранняя остановка на эпохе {ep + 1}")
            break
    head.load_state_dict(best_state)
    return head, hist


# ======================================================================
# Графики
# ======================================================================
def plot_weight_curve(ws, eers, w_best, path: Path, title: str):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(ws, 100 * np.array(eers), "o-")
    ax.axvline(w_best, color="k", ls="--", label=f"w* = {w_best:.2f}")
    ax.set(xlabel="вес отпечатка w   (0 = только радужка, 1 = только отпечаток)", ylabel="val EER, %", title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_group_bars(groups: dict, path: Path, title: str):
    if not groups:
        return
    plt = _plt()
    names = list(groups)
    meths = ["fp_only", "iris_only", "score_weighted", "feat_learned"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(names)), 4))
    for k, m in enumerate(meths):
        ax.bar(x + (k - 1.5) * 0.2, [100 * groups[g].get(m, {}).get("eer", np.nan) for g in names], 0.2,
               label=LABELS[m])
    ax.set_xticks(x, names, rotation=25)
    ax.set_ylabel("EER, % (проба против эталона)")
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ======================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Слияние отпечатка и радужки (неделя 8)")
    ap.add_argument("--corpus", required=True, choices=("multimodal_sd302", "multimodal_socofing"))
    ap.add_argument("--fp-run", required=True, help="папка прогона модели отпечатков (runs/<имя>)")
    ap.add_argument("--iris-run", required=True, help="папка прогона модели радужки")
    ap.add_argument("--name", default="", help="имя папки результата (по умолчанию fusion_<корпус>)")
    ap.add_argument("--val-impostors", type=int, default=200000)
    ap.add_argument("--test-impostors", type=int, default=1000000)
    ap.add_argument("--epochs", type=int, default=80, help="эпох для feat_learned")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--s", type=float, default=30.0)
    ap.add_argument("--m", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    fp_run = (ROOT / args.fp_run) if not Path(args.fp_run).is_absolute() else Path(args.fp_run)
    ir_run = (ROOT / args.iris_run) if not Path(args.iris_run).is_absolute() else Path(args.iris_run)
    fp_run, ir_run = fp_run.resolve(), ir_run.resolve()
    os.chdir(ROOT)
    name = args.name or f"fusion_{args.corpus}"
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / "fusion.log")
    log = lambda s: print(s, flush=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"\n##### {name}  ({datetime.now():%Y-%m-%d %H:%M})  корпус {args.corpus}")

    _, fp_cfg = load_run(fp_run, "cpu")
    _, ir_cfg = load_run(ir_run, "cpu")
    want_fp = {"multimodal_sd302": "fingerprint_sd302", "multimodal_socofing": "fingerprint_socofing"}[args.corpus]
    if fp_cfg["modality"] != want_fp:
        raise SystemExit(f"--fp-run обучен на {fp_cfg['modality']}, а корпус {args.corpus} требует {want_fp}")
    if ir_cfg["modality"] != "iris":
        raise SystemExit(f"--iris-run обучен на {ir_cfg['modality']}, нужна радужка")
    fp_col = "fp_image_path_clahe" if fp_cfg.get("fp_input", "clahe") == "clahe" else "fp_image_path"
    log(f"Отпечаток: {fp_run.name} ({fp_cfg['backbone']}), радужка: {ir_run.name} ({ir_cfg['backbone']}); "
        f"устройство {device}")

    rows = {s: read_csv(OUTPUT_SPLITS / args.corpus / f"{s}.csv") for s in SPLITS}
    log("Эмбеддинги:")
    efp = embed(fp_run, [r[fp_col] for s in SPLITS for r in rows[s]], FP_SIZE, device,
                out / f"emb_fp_{fp_run.name}.npz", 64, args.num_workers)
    eir = embed(ir_run, [r["iris_image_path"] for s in SPLITS for r in rows[s]], IRIS_SIZE, device,
                out / f"emb_iris_{ir_run.name}.npz", 64, args.num_workers)
    sp = {s: Split(rows[s], efp, eir, fp_col) for s in SPLITS}
    for s in SPLITS:
        log(f"  {s:5s}: {len(sp[s].y)} пар снимков, {int(sp[s].y.max()) + 1} виртуальных субъектов")

    pairs = {"val": make_pairs(sp["val"], args.val_impostors, args.seed),
             "test": make_pairs(sp["test"], args.test_impostors, args.seed)}

    # --- z-нормализация и вес по val
    (ga, gb), (ia, ib) = pairs["val"]
    vfp = np.r_[rowdot(sp["val"].Efp, ga, gb), rowdot(sp["val"].Efp, ia, ib)]
    vir = np.r_[rowdot(sp["val"].Eir, ga, gb), rowdot(sp["val"].Eir, ia, ib)]
    z = {"fp_mu": float(vfp.mean()), "fp_sd": float(vfp.std()), "ir_mu": float(vir.mean()), "ir_sd": float(vir.std())}
    ws = np.round(np.arange(0, 1.0001, 0.05), 2)
    sg = (rowdot(sp["val"].Efp, ga, gb), rowdot(sp["val"].Eir, ga, gb))
    si = (rowdot(sp["val"].Efp, ia, ib), rowdot(sp["val"].Eir, ia, ib))
    eers = [verification(combine("score_weighted", *sg, None, z, w), combine("score_weighted", *si, None, z, w))[0]["eer"]
            for w in ws]
    tied = ws[np.asarray(eers) <= min(eers) + 1e-6]
    w_best = float(tied[len(tied) // 2])
    log(f"Слияние оценок: z-нормализация по val; лучший вес отпечатка w* = {w_best:.2f} "
        f"(val EER {100 * min(eers):.2f}%)")
    plot_weight_curve(ws, eers, w_best, out / "score_weight_curve.png", f"{args.corpus}: вес отпечатка")

    # --- слияние признаков с обучением
    log("Слияние признаков с обучением (feat_learned):")
    head, hist = train_learned(sp["train"], sp["val"], pairs["val"], args, device, log)
    for s in ("val", "test"):
        sp[s].Efu = fused_embeddings(head, sp[s], device)
    torch.save({"head": head.state_dict(), "args": vars(args), "fp_run": str(fp_run), "iris_run": str(ir_run)},
               out / "fusion_head.pt")

    # --- оценка всех методов на одних и тех же парах
    results, curves, thr = {}, {}, {}
    for s in ("val", "test"):
        (ga, gb), (ia, ib) = pairs[s]
        g, i = sp[s].pair_scores(ga, gb), sp[s].pair_scores(ia, ib)
        for m in METHODS:
            gen, imp = combine(m, *g, z, w_best), combine(m, *i, z, w_best)
            res, curve = verification(gen, imp)
            if s == "val":
                thr[m] = res["eer_threshold"]
            else:
                res.update({f"op_{k}": v for k, v in operating_point(gen, imp, thr[m], args.seed).items()})
                res.update({f"gp_{k}": v for k, v in (identification(sp[s], m, z, w_best) or {}).items()})
                curves[LABELS[m]] = curve
            results.setdefault(m, {})[s] = res

    # --- разбивка test по группам (проба против эталона субъекта)
    te = sp["test"]
    key = "fp_subset" if args.corpus == "multimodal_socofing" else "fp_device"
    grp = defaultdict(list)
    for idx, r in enumerate(te.rows):
        if r.get(key):
            grp[f"fp:{r[key]}"].append(idx)
        try:
            o = float(r.get("iris_occlusion", ""))
            grp["iris occl " + ("<0.2" if o < 0.2 else "0.2-0.4" if o <= 0.4 else ">0.4")].append(idx)
        except ValueError:
            pass
    groups = {}
    for gname in sorted(grp):
        for m in METHODS:
            r = identification(te, m, z, w_best, grp[gname])
            if r:
                groups.setdefault(gname, {})[m] = r

    # --- время отклика: сумма двух моделей (голова слияния пренебрежимо мала)
    timing = {}
    for tag, run in (("fp", fp_run), ("iris", ir_run)):
        mj = run / "metrics.json"
        if mj.exists():
            timing[tag] = json.load(open(mj, encoding="utf-8")).get("timing", {})
    t_cpu = sum(t.get("cpu_ms_per_image", 0) for t in timing.values())
    t_gpu = sum(t.get("gpu_ms_per_image", 0) for t in timing.values())

    # --- таблица
    hdr = ("| Метод | val EER | test EER | AUC | TAR@FAR=0,1% | FAR | FRR | Accuracy | Precision | Recall | F1 "
           "| Rank-1 | Rank-5 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [hdr]
    for m in METHODS:
        v, t = results[m]["val"], results[m]["test"]
        p = lambda x: "—" if x is None else f"{100 * x:.2f}%"  # noqa: E731
        lines.append(f"| {LABELS[m]} | {p(v['eer'])} | **{p(t['eer'])}** | {t['auc']:.4f} | "
                     f"{p(t.get('tar_at_far_1e-3'))} | {p(t['op_far'])} | {p(t['op_frr'])} | {p(t['op_accuracy'])} | "
                     f"{p(t['op_precision'])} | {p(t['op_recall'])} | {p(t['op_f1'])} | "
                     f"{p(t.get('gp_rank1'))} | {p(t.get('gp_rank5'))} |")
    table = "\n".join(lines)
    gl = ["", "**Test по группам (EER, проба против эталона субъекта):**", "",
          "| Группа | проб | " + " | ".join(LABELS[m] for m in METHODS) + " |",
          "|---|---|" + "---|" * len(METHODS)]
    for gname, gm in groups.items():
        n = next(iter(gm.values()))["n_probes"]
        gl.append(f"| {gname} | {n} | " + " | ".join(f"{100 * gm[m]['eer']:.2f}%" if m in gm else "—"
                                                    for m in METHODS) + " |")
    md = (f"# Слияние: {args.corpus}\n\nОтпечаток: `{fp_run.name}`, радужка: `{ir_run.name}`; "
          f"w* = {w_best:.2f}; z-параметры по val: {json.dumps({k: round(v, 4) for k, v in z.items()})}\n"
          f"Время отклика (две модели последовательно): CPU {t_cpu:.1f} мс, GPU {t_gpu:.1f} мс на пару снимков\n\n"
          + table + "\n" + "\n".join(gl) + "\n")
    (out / "summary.md").write_text(md, encoding="utf-8")
    log("\n" + md)

    plot_roc_det(curves, out / "roc_det.png", f"{args.corpus}: test")
    plot_group_bars(groups, out / "groups_test.png", f"{args.corpus}: test по группам")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot([h["epoch"] for h in hist], [100 * h["val_eer"] for h in hist], "o-", ms=3)
    ax.set(xlabel="эпоха", ylabel="val EER, %", title="feat_learned: обучение проекции")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "learned_curve.png", dpi=130)
    plt.close(fig)

    metrics = {"corpus": args.corpus, "fp_run": fp_run.name, "iris_run": ir_run.name, "w_best": w_best, "z": z,
               "results": results, "test_groups": groups, "timing_ms": {"cpu": t_cpu, "gpu": t_gpu},
               "learned_history": hist, "args": vars(args), **git_info()}
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=float)

    if not args.no_mlflow:
        tr = Tracker({"mlflow": True, "mlflow_uri": "", "mlflow_experiment": "fusion", "run_name": name})
        tr.params({"corpus": args.corpus, "fp_run": fp_run.name, "iris_run": ir_run.name, "w_best": w_best,
                   **{k: v for k, v in vars(args).items() if k not in ("corpus", "fp_run", "iris_run")}})
        tr.tags(git_info())
        tr.metrics({f"{m}_{s}_eer": results[m][s]["eer"] for m in METHODS for s in ("val", "test")})
        for f in ("summary.md", "metrics.json", "roc_det.png", "groups_test.png", "score_weight_curve.png",
                  "learned_curve.png", "fusion.log"):
            tr.artifact(out / f)
        tr.end()
    log(f"Результаты: {out}")


if __name__ == "__main__":
    main()
