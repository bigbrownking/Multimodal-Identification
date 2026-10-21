import argparse
import copy
import csv
import json
import math
import os
import random
import subprocess
import sys
import mlflow
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler.step\(\)` before")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.datasets import BiometricDataset, make_loader  # noqa: E402

RUNS_DIR = ROOT / "runs"

DEFAULTS = {
    "modality": "fingerprint",
    "backbone": "resnet50",
    "pretrained": True,
    "fp_input": "clahe",
    "embedding_dim": 512,
    "dropout": 0.0,
    "arcface_s": 30.0,
    "arcface_m": 0.5,
    "margin_warmup_epochs": 3,
    "epochs": 30,
    "batch_size": 32,
    "lr": 1.0e-3,
    "backbone_lr_mult": 0.1,
    "weight_decay": 1.0e-2,
    "warmup_epochs": 1,
    "min_lr_ratio": 0.01,
    "grad_clip": 5.0,
    "amp": True,
    "num_workers": 4,
    "seed": 42,
    "patience": 8,
    "val_impostors": 200000,
    "test_impostors": 1000000,
    "max_train_steps": 0,
    "max_eval_samples": 0,
    "timing_iters": 50,
    "mlflow": True,
    "mlflow_uri": "",
    "mlflow_experiment": "unimodal",
    "run_name": "",
}

INPUT_SHAPE = {"fingerprint": (3, 224, 224), "iris": (3, 64, 512)}


# ======================================================================
# Конфиг, логирование, воспроизводимость
# ======================================================================
def coerce(cfg: dict) -> dict:
    out = dict(cfg)
    for k, v in cfg.items():
        if k not in DEFAULTS:
            raise SystemExit(f"Неизвестный параметр конфига: {k}")
        d = DEFAULTS[k]
        if isinstance(d, bool):
            out[k] = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
        elif isinstance(d, int):
            out[k] = int(float(v))
        elif isinstance(d, float):
            out[k] = float(v)
        else:
            out[k] = "" if v is None else str(v)
    return out


def load_config(path: str | None, overrides: list[str]) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    for kv in overrides:
        k, _, v = kv.partition("=")
        cfg[k.strip()] = yaml.safe_load(v)
    cfg = coerce(cfg)
    assert cfg["modality"] in ("fingerprint", "iris"), cfg["modality"]
    return cfg


class Tee:
    def __init__(self, path: Path):
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        try:
            self.stdout.write(s)
        except UnicodeEncodeError:
            self.stdout.write(s.encode("ascii", "replace").decode())
        self.file.write(s)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_info() -> dict:
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                    stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                             text=True, stderr=subprocess.DEVNULL).strip())
        return {"git_commit": h, "git_dirty": str(dirty)}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": "unknown"}


class Tracker:
    def __init__(self, cfg: dict, run_id: str | None = None):
        self.ml, self.run_id = None, None
        if not cfg["mlflow"]:
            return
        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            mlflow.set_tracking_uri(cfg["mlflow_uri"] or "sqlite:///" + (ROOT / "mlflow.db").as_posix())
            mlflow.set_experiment(cfg["mlflow_experiment"])
            run = mlflow.start_run(run_id=run_id, run_name=cfg["run_name"])
            self.ml, self.run_id = mlflow, run.info.run_id
            print(f"MLflow: эксперимент '{cfg['mlflow_experiment']}', run_id {self.run_id}")
        except Exception as e:
            print(f"[!] MLflow недоступен ({e}) — пишу только в папку прогона")

    def _call(self, fn, *a, **kw):
        if self.ml is None:
            return
        try:
            fn(*a, **kw)
        except Exception as e:
            print(f"[!] MLflow: {e}")

    def params(self, d: dict):
        self._call(self.ml and self.ml.log_params, {k: str(v)[:250] for k, v in d.items()})

    def tags(self, d: dict):
        self._call(self.ml and self.ml.set_tags, d)

    def metrics(self, d: dict, step: int | None = None):
        d = {k: float(v) for k, v in d.items() if v is not None and np.isfinite(v)}
        self._call(self.ml and self.ml.log_metrics, d, step=step)

    def artifact(self, path: Path, sub: str | None = None):
        if Path(path).exists():
            self._call(self.ml and self.ml.log_artifact, str(path), sub)

    def end(self, status: str = "FINISHED"):
        self._call(self.ml and self.ml.end_run, status)


# ======================================================================
# Модель
# ======================================================================
class EmbeddingNet(nn.Module):
    def __init__(self, backbone: str, dim: int = 512, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, global_pool="avg")
        self.backbone.eval()
        with torch.no_grad():
            feat_dim = self.backbone(torch.zeros(1, 3, 64, 64)).shape[1]
        self.backbone.train()
        self.neck = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, dim, bias=False),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return self.neck(self.backbone(x))


class ArcFace(nn.Module):
    def __init__(self, dim: int, n_classes: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.empty(n_classes, dim))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor, m: float | None = None):
        m = self.m if m is None else m
        cos = F.linear(F.normalize(emb.float()), F.normalize(self.weight.float()))
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
        if m <= 0:
            return cos * self.s, cos
        c = cos.gather(1, labels[:, None])
        s = torch.sqrt(1.0 - c * c)
        phi = c * math.cos(m) - s * math.sin(m)
        phi = torch.where(c > math.cos(math.pi - m), phi, c - math.sin(math.pi - m) * m)
        return cos.scatter(1, labels[:, None], phi) * self.s, cos


def param_groups(model: EmbeddingNet, head: ArcFace, cfg: dict) -> list[dict]:
    groups = []
    for mod, lr in [(model.backbone, cfg["lr"] * cfg["backbone_lr_mult"]),
                    (nn.ModuleList([model.neck, head]), cfg["lr"])]:
        decay = [p for p in mod.parameters() if p.ndim > 1]
        no_decay = [p for p in mod.parameters() if p.ndim <= 1]
        groups += [{"params": decay, "lr": lr, "weight_decay": cfg["weight_decay"]},
                   {"params": no_decay, "lr": lr, "weight_decay": 0.0}]
    return groups


def lr_lambda(total: int, warm: int, min_ratio: float):
    def f(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    return f


# ======================================================================
# Метрики верификации и идентификации
# ======================================================================
def nonfinite_state(model: nn.Module) -> list[str]:
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    bad += [n for n, b in model.named_buffers() if b.is_floating_point() and not torch.isfinite(b).all()]
    return bad


@torch.no_grad()
def extract(model: nn.Module, loader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        e = model(x.to(device, non_blocking=True)).float()
        out.append(F.normalize(e, dim=1).cpu())
    E = torch.cat(out).numpy()
    bad = ~np.isfinite(E).all(1)
    if bad.any():
        state = nonfinite_state(model)
        raise RuntimeError(
            f"NaN/inf в {int(bad.sum())} из {len(E)} эмбеддингов. "
            + (f"Испорчены веса/буферы модели: {state[:5]} (всего {len(state)}) — "
               f"обучение в fp16 разошлось, запустите с --set amp=false"
               if state else "Веса модели в порядке — проблема во входных данных или вычислениях на GPU"))
    return E


def genuine_pairs(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    by = defaultdict(list)
    for i, c in enumerate(labels):
        by[c].append(i)
    a, b = [], []
    for ids in by.values():
        if len(ids) < 2:
            continue
        ids = np.asarray(ids)
        i, j = np.triu_indices(len(ids), 1)
        a.append(ids[i])
        b.append(ids[j])
    return np.concatenate(a), np.concatenate(b)


def impostor_pairs(labels: np.ndarray, n: int, rng: np.random.Generator):
    a = rng.integers(0, len(labels), int(n * 1.1) + 100)
    b = rng.integers(0, len(labels), len(a))
    keep = labels[a] != labels[b]
    return a[keep][:n], b[keep][:n]


def pair_scores(E: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 65536) -> np.ndarray:
    out = np.empty(len(a), np.float32)
    for s in range(0, len(a), chunk):
        out[s:s + chunk] = np.einsum("ij,ij->i", E[a[s:s + chunk]], E[b[s:s + chunk]])
    return out


def compute_eer(far: np.ndarray, frr: np.ndarray, thr: np.ndarray) -> tuple[float, float]:
    i = int(np.argmax(far >= frr))
    if i == 0:
        return float(far[0]), float(thr[0])
    x0, x1 = frr[i - 1] - far[i - 1], frr[i] - far[i]
    t = x0 / (x0 - x1) if x0 != x1 else 0.0
    eer = far[i - 1] + t * (far[i] - far[i - 1])
    th = thr[i - 1] + t * (thr[i] - thr[i - 1])
    if not np.isfinite(th):
        th = thr[i]
    return float(eer), float(th)


def verification(gen: np.ndarray, imp: np.ndarray) -> tuple[dict, dict]:
    from sklearn.metrics import auc, roc_curve
    y = np.r_[np.ones(len(gen)), np.zeros(len(imp))]
    far, tar, thr = roc_curve(y, np.r_[gen, imp])
    frr = 1 - tar
    eer, eer_thr = compute_eer(far, frr, thr)
    res = {"eer": eer, "eer_threshold": eer_thr, "auc": float(auc(far, tar)),
           "n_genuine": int(len(gen)), "n_impostor": int(len(imp))}
    for name, target in [("1e-2", 1e-2), ("1e-3", 1e-3), ("1e-4", 1e-4)]:
        if len(imp) * target >= 10:
            res[f"tar_at_far_{name}"] = float(np.interp(target, far, tar))
    return res, {"far": far, "frr": frr, "thr": thr}


def verify_split(E: np.ndarray, labels: np.ndarray, n_imp: int, seed: int):
    a, b = genuine_pairs(labels)
    gen = pair_scores(E, a, b)
    ia, ib = impostor_pairs(labels, n_imp, np.random.default_rng(seed))
    imp = pair_scores(E, ia, ib)
    res, curve = verification(gen, imp)
    return res, curve, gen, imp


def operating_point(gen: np.ndarray, imp: np.ndarray, thr: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    imp_b = rng.choice(imp, size=min(len(gen), len(imp)), replace=False)
    tp, fn = int((gen >= thr).sum()), int((gen < thr).sum())
    fp, tn = int((imp_b >= thr).sum()), int((imp_b < thr).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"threshold": float(thr),
            "far": float((imp >= thr).mean()), "frr": float((gen < thr).mean()),
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12)}


def gallery_index(rows: list[dict], labels: np.ndarray) -> dict:
    order = sorted(range(len(rows)), key=lambda i: (rows[i].get("dataset", "") != "Real",
                                                    rows[i].get("sample_id", "")))
    gal = {}
    for i in order:
        gal.setdefault(labels[i], i)
    return gal


def gallery_probe_eval(E, labels, gal: dict, probe_idx, n_imp: int, seed: int) -> dict | None:
    gal_idx = np.array(list(gal.values()))
    gal_cls = labels[gal_idx]
    pos = {c: k for k, c in enumerate(gal_cls)}
    gal_set = set(gal_idx.tolist())
    probe_idx = np.array([i for i in probe_idx if labels[i] in pos and i not in gal_set], dtype=int)
    if len(probe_idx) < 5:
        return None
    own = np.array([pos[labels[i]] for i in probe_idx])
    S = E[probe_idx] @ E[gal_idx].T
    gen = S[np.arange(len(own)), own]
    S[np.arange(len(own)), own] = -np.inf
    rank1 = float(np.mean(gen > S.max(1)))
    k = min(5, S.shape[1])
    kth = -np.partition(-S, k - 2, axis=1)[:, k - 2] if k >= 2 else np.full(len(own), -np.inf)
    rank5 = float(np.mean(gen > kth))
    rng = np.random.default_rng(seed)
    pi = rng.integers(0, len(own), n_imp)
    gj = rng.integers(0, len(gal_idx), n_imp)
    keep = gj != own[pi]
    imp = S[pi[keep], gj[keep]]
    res, _ = verification(gen, imp)
    return {"rank1": rank1, "rank5": rank5, "eer": res["eer"], "n_probes": int(len(own))}


def group_breakdown(E, labels, rows, modality: str, n_imp: int, seed: int) -> dict:
    gal = gallery_index(rows, labels)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if modality == "fingerprint":
            ds = r.get("dataset", "")
            if ds and ds != "Real":
                groups[f"level/{ds}"].append(i)
                alt = r.get("alteration", "")
                if alt and alt.lower() != "none":
                    groups[f"alteration/{alt}"].append(i)
        else:
            try:
                occ = float(r.get("occlusion_ratio", ""))
            except ValueError:
                continue
            g = "<0.2" if occ < 0.2 else "0.2-0.4" if occ <= 0.4 else ">0.4"
            groups[f"occlusion/{g}"].append(i)
            if r.get("dataset"):
                groups[f"dataset/{r['dataset']}"].append(i)
    out = {}
    for name in sorted(groups):
        res = gallery_probe_eval(E, labels, gal, groups[name], n_imp, seed)
        if res:
            out[name] = res
    return out


# ======================================================================
# Графики
# ======================================================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_history(history: list[dict], path: Path, title: str) -> None:
    if not history:
        return
    plt = _plt()
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-")
    ax[0].set_title("Train loss (ArcFace)")
    ax[1].plot(ep, [100 * h["train_acc"] for h in history], "o-")
    ax[1].set_title("Train accuracy, % (без отступа)")
    ax[2].plot(ep, [100 * h["val_eer"] for h in history], "o-", color="C3")
    ax[2].set_title("Val EER, %")
    ax[3].plot(ep, [100 * h["val_rank1"] for h in history], "o-", color="C2")
    ax[3].set_title("Val rank-1, %")
    for a in ax:
        a.set_xlabel("эпоха")
        a.grid(alpha=0.3)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc_det(curves: dict, path: Path, title: str) -> None:
    from scipy.stats import norm
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ticks = np.array([0.0001, 0.001, 0.01, 0.05, 0.2, 0.5])
    for name, c in curves.items():
        far, frr = c["far"], c["frr"]
        m = far > 0
        ax[0].semilogx(far[m], 1 - frr[m], label=name)
        e = np.clip(far, 1e-6, 1 - 1e-6), np.clip(frr, 1e-6, 1 - 1e-6)
        ax[1].plot(norm.ppf(e[0]), norm.ppf(e[1]), label=name)
    ax[0].set(xlabel="FAR", ylabel="TAR = 1 - FRR", title="ROC", xlim=(1e-5, 1))
    ax[1].set(xlabel="FAR", ylabel="FRR", title="DET")
    ax[1].set_xticks(norm.ppf(ticks), [f"{100 * t:g}%" for t in ticks])
    ax[1].set_yticks(norm.ppf(ticks), [f"{100 * t:g}%" for t in ticks])
    ax[1].set_xlim(norm.ppf(1e-5), norm.ppf(0.6))
    ax[1].set_ylim(norm.ppf(1e-4), norm.ppf(0.6))
    ax[1].plot(norm.ppf([1e-5, 0.6]), norm.ppf([1e-5, 0.6]), "k:", lw=0.8, label="EER")
    for a in ax:
        a.grid(alpha=0.3, which="both")
        a.legend()
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scores(gen, imp, thr, path: Path, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-0.5, 1.0, 120)
    ax.hist(imp, bins=bins, density=True, alpha=0.6, label=f"разные ({len(imp):,})")
    ax.hist(gen, bins=bins, density=True, alpha=0.6, label=f"тот же ({len(gen):,})")
    ax.axvline(thr, color="k", ls="--", label=f"порог с val = {thr:.3f}")
    ax.set(xlabel="косинусное сходство", ylabel="плотность", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_groups(groups: dict, path: Path, title: str) -> None:
    if not groups:
        return
    plt = _plt()
    names = list(groups)
    fig, ax = plt.subplots(1, 2, figsize=(max(8, 1.2 * len(names)), 4))
    ax[0].bar(names, [100 * groups[n]["eer"] for n in names], color="C3")
    ax[0].set_title("EER, % (проба против эталона)")
    ax[1].bar(names, [100 * groups[n]["rank1"] for n in names], color="C2")
    ax[1].set_title("Rank-1, %")
    for a in ax:
        a.tick_params(axis="x", rotation=35)
        a.grid(alpha=0.3, axis="y")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ======================================================================
# Обучение
# ======================================================================
def make_eval_set(cfg: dict, split: str, nw: int):
    ds = BiometricDataset(cfg["modality"], split, train=False, fp_input=cfg["fp_input"])
    if cfg["max_eval_samples"] and len(ds) > cfg["max_eval_samples"]:
        keep = sorted(range(len(ds)), key=lambda i: ds.classes[i])[:cfg["max_eval_samples"]]
        ds.paths = [ds.paths[i] for i in keep]
        ds.classes = [ds.classes[i] for i in keep]
        ds.labels = [ds.labels[i] for i in keep]
        ds.rows = [ds.rows[i] for i in keep]
    loader = make_loader(ds, batch_size=cfg["batch_size"] * 2, shuffle=False, num_workers=nw)
    return ds, loader


def evaluate_val(model, ds, loader, device, cfg) -> dict:
    E = extract(model, loader, device)
    labels = np.asarray(ds.labels)
    res, _, _, _ = verify_split(E, labels, cfg["val_impostors"], cfg["seed"])
    gal = gallery_index(ds.rows, labels)
    gp = gallery_probe_eval(E, labels, gal, range(len(labels)), 100000, cfg["seed"]) or {}
    return {"val_eer": res["eer"], "val_auc": res["auc"],
            "val_tar_at_far_1e-3": res.get("tar_at_far_1e-3"), "val_rank1": gp.get("rank1", 0.0)}


def train_one_epoch(model, head, loader, opt, sched, scaler, device, cfg, margin) -> dict:
    model.train()
    head.train()
    use_amp = cfg["amp"] and device.type == "cuda"
    params = list(model.parameters()) + list(head.parameters())
    tot, correct, n, skipped = 0.0, 0, 0, 0
    steps = len(loader) if not cfg["max_train_steps"] else min(len(loader), cfg["max_train_steps"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    t_data = t_gpu = 0.0
    t_prev = time.perf_counter()
    for step, (x, y) in enumerate(loader):
        if step >= steps:
            break
        t_got = time.perf_counter()
        if step > 0:
            t_data += t_got - t_prev
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            emb = model(x)
        logits, cos = head(emb, y, margin)
        loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            if skipped >= 20 and n == 0:
                raise RuntimeError("Первые 20 шагов дали NaN/inf loss — обучение не идёт. "
                                   "Если включён AMP, запустите с --set amp=false")
            t_prev = time.perf_counter()
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg["grad_clip"] > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        bs = y.size(0)
        tot += loss.item() * bs
        correct += (cos.argmax(1) == y).sum().item()
        n += bs
        t_prev = time.perf_counter()
        if step > 0:
            t_gpu += t_prev - t_got
        if (step + 1) % 100 == 0 or step + 1 == steps:
            el = time.perf_counter() - t0
            k = max(step, 1)
            mem = (f"  VRAM пик {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} ГБ"
                   if device.type == "cuda" else "")
            print(f"    шаг {step + 1:5d}/{steps}  loss {tot / max(n, 1):.4f}  "
                  f"acc {100 * correct / max(n, 1):5.1f}%  {el / (step + 1) * 1000:.0f} мс/шаг "
                  f"(ожидание данных {t_data / k * 1000:.0f} + расчёт {t_gpu / k * 1000:.0f} мс){mem}",
                  flush=True)
    if skipped:
        print(f"    [!] пропущено шагов с NaN/inf loss: {skipped} — если их много, запустите с --set amp=false")
    bad = nonfinite_state(model)
    if bad:
        raise RuntimeError(f"После эпохи NaN/inf в {len(bad)} весах/буферах (напр. {bad[:3]}). "
                           f"Обучение в fp16 разошлось — запустите заново с --set amp=false")
    return {"train_loss": tot / max(n, 1), "train_acc": correct / max(n, 1)}


def save_history(history: list[dict], path: Path) -> None:
    if not history:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def train(cfg: dict, run_dir: Path, resume: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    nw = cfg["num_workers"]
    if cfg["amp"] and device.type == "cuda" and "GTX 16" in torch.cuda.get_device_name(0):
        cfg["amp"] = False
        print("[i] GTX 16xx: AMP (fp16) выключен — на этих картах fp16 даёт NaN; обучение в fp32")

    train_ds = BiometricDataset(cfg["modality"], "train", fp_input=cfg["fp_input"])
    train_loader = make_loader(train_ds, batch_size=cfg["batch_size"], num_workers=nw)
    val_ds, val_loader = make_eval_set(cfg, "val", nw)
    print(f"Данные: train {len(train_ds)} образцов / {train_ds.num_classes} классов, "
          f"val {len(val_ds)} / {val_ds.num_classes}")

    model = EmbeddingNet(cfg["backbone"], cfg["embedding_dim"], cfg["pretrained"], cfg["dropout"]).to(device)
    head = ArcFace(cfg["embedding_dim"], train_ds.num_classes, cfg["arcface_s"], cfg["arcface_m"]).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Модель: {cfg['backbone']} -> {cfg['embedding_dim']}, {n_params:.1f} M параметров; "
          f"ArcFace {train_ds.num_classes} классов, s={cfg['arcface_s']}, m={cfg['arcface_m']}; "
          f"устройство {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    opt = torch.optim.AdamW(param_groups(model, head, cfg))
    steps = len(train_loader) if not cfg["max_train_steps"] else min(len(train_loader), cfg["max_train_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(cfg["epochs"] * steps, cfg["warmup_epochs"] * steps, cfg["min_lr_ratio"]))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")

    history, start, best_eer, best_epoch, run_id = [], 0, float("inf"), -1, None
    last_pt = run_dir / "last.pt"
    if resume and last_pt.exists():
        ck = torch.load(last_pt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        history, start = ck["history"], ck["epoch"] + 1
        best_eer, best_epoch, run_id = ck["best_eer"], ck["best_epoch"], ck.get("mlflow_run_id")
        print(f"Продолжаю с эпохи {start + 1} (лучшая val EER {100 * best_eer:.2f}% на эпохе {best_epoch + 1})")

    tracker = Tracker(cfg, run_id)
    if start == 0:
        tracker.params({**cfg, "train_samples": len(train_ds), "train_classes": train_ds.num_classes,
                        "val_samples": len(val_ds), "params_m": round(n_params, 2)})
        tracker.tags({**git_info(), "device": str(device), "torch": torch.__version__})
        tracker.artifact(run_dir / "config.yaml")

    status = "FINISHED"
    try:
        for epoch in range(start, cfg["epochs"]):
            w = cfg["margin_warmup_epochs"]
            margin = cfg["arcface_m"] * min(1.0, epoch / w) if w > 0 else cfg["arcface_m"]
            print(f"\n=== Эпоха {epoch + 1}/{cfg['epochs']}  lr {sched.get_last_lr()[-1]:.2e}  m {margin:.2f}")
            t0 = time.perf_counter()
            tr = train_one_epoch(model, head, train_loader, opt, sched, scaler, device, cfg, margin)
            t_train = time.perf_counter() - t0
            va = evaluate_val(model, val_ds, val_loader, device, cfg)
            rec = {"epoch": epoch + 1, **tr, **va, "lr": sched.get_last_lr()[-1], "margin": margin,
                   "epoch_time_s": time.perf_counter() - t0, "train_time_s": t_train}
            history.append(rec)
            improved = va["val_eer"] < best_eer
            if improved:
                best_eer, best_epoch = va["val_eer"], epoch
                torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                            "val": va, "num_train_classes": train_ds.num_classes}, run_dir / "best.pt")
            print(f"  train loss {tr['train_loss']:.4f}, acc {100 * tr['train_acc']:.1f}% | "
                  f"val EER {100 * va['val_eer']:.2f}%, AUC {va['val_auc']:.4f}, "
                  f"rank-1 {100 * va['val_rank1']:.1f}% | {rec['epoch_time_s'] / 60:.1f} мин"
                  + ("  <- лучшая" if improved else ""), flush=True)
            tracker.metrics({k: v for k, v in rec.items() if k != "epoch"}, step=epoch + 1)
            save_history(history, run_dir / "history.csv")
            plot_history(history, run_dir / "learning_curves.png",
                         f"{cfg['modality']} / {cfg['backbone']}")
            torch.save({"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                        "history": history, "best_eer": best_eer, "best_epoch": best_epoch,
                        "mlflow_run_id": tracker.run_id}, last_pt)
            if epoch - best_epoch >= cfg["patience"]:
                print(f"\nРанняя остановка: {cfg['patience']} эпох без улучшения val EER")
                break
    except KeyboardInterrupt:
        status = "KILLED"
        print("\n[!] Остановлено вручную. Продолжить: python training/train.py --resume", run_dir)
        tracker.end(status)
        raise
    except Exception:
        tracker.end("FAILED")
        raise

    print(f"\nЛучшая эпоха {best_epoch + 1}: val EER {100 * best_eer:.2f}%")
    for f in ("history.csv", "learning_curves.png", "train.log"):
        tracker.artifact(run_dir / f)
    del train_loader, val_loader
    final_evaluation(cfg, run_dir, device, tracker, history)
    tracker.end(status)


# ======================================================================
# Финальная оценка
# ======================================================================
def inference_time(model: nn.Module, shape: tuple, device: torch.device, iters: int, amp: bool) -> dict:
    out = {}
    x = torch.randn(1, *shape)
    with torch.inference_mode():
        if device.type == "cuda":
            m, xg = model.to(device).eval(), x.to(device)
            for _ in range(10):
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    m(xg)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(iters):
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    m(xg)
            torch.cuda.synchronize()
            out["gpu_ms_per_image"] = (time.perf_counter() - t) / iters * 1000
        mc = copy.deepcopy(model).cpu().eval()
        for _ in range(3):
            mc(x)
        n = max(5, iters // 3)
        t = time.perf_counter()
        for _ in range(n):
            mc(x)
        out["cpu_ms_per_image"] = (time.perf_counter() - t) / n * 1000
    return out


def save_embeddings(path: Path, E: np.ndarray, ds: BiometricDataset) -> None:
    cols = {k: np.array([r.get(k, "") for r in ds.rows]) for k in
            ("subject_id", "person_id", "sample_id", "image_path", "dataset",
             "alteration", "eye", "quality_score", "occlusion_ratio")}
    np.savez_compressed(path, emb=E.astype(np.float16), **cols)


def final_evaluation(cfg: dict, run_dir: Path, device: torch.device, tracker: Tracker,
                     history: list[dict] | None = None) -> dict:
    print("\n" + "=" * 60 + "\nФИНАЛЬНАЯ ОЦЕНКА (best.pt)\n" + "=" * 60)
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model = EmbeddingNet(cfg["backbone"], cfg["embedding_dim"], pretrained=False, dropout=cfg["dropout"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    nw, n_imp, seed = cfg["num_workers"], cfg["test_impostors"], cfg["seed"]
    title = f"{cfg['modality']} / {cfg['backbone']}"

    results, curves, extra = {}, {}, {}
    for split in ("val", "test"):
        ds, loader = make_eval_set(cfg, split, nw)
        E = extract(model, loader, device)
        del loader
        labels = np.asarray(ds.labels)
        res, curve, gen, imp = verify_split(E, labels, n_imp, seed)
        gal = gallery_index(ds.rows, labels)
        res.update({f"gp_{k}": v for k, v in
                    (gallery_probe_eval(E, labels, gal, range(len(labels)), n_imp, seed) or {}).items()})
        results[split], curves[split] = res, curve
        extra[split] = (E, labels, ds, gen, imp)
        save_embeddings(run_dir / f"embeddings_{split}.npz", E, ds)
        print(f"  {split:4s}: {len(ds)} образцов, {ds.num_classes} классов | EER {100 * res['eer']:.2f}% "
              f"(порог {res['eer_threshold']:.3f}), AUC {res['auc']:.4f}, "
              f"TAR@FAR=0.1% {100 * res.get('tar_at_far_1e-3', float('nan')):.1f}%, "
              f"rank-1 {100 * res.get('gp_rank1', 0):.1f}%")

    thr = results["val"]["eer_threshold"]
    _, _, ds_t, gen_t, imp_t = extra["test"]
    op = operating_point(gen_t, imp_t, thr, seed)
    print(f"  test при пороге с val ({thr:.3f}): FAR {100 * op['far']:.2f}%, FRR {100 * op['frr']:.2f}%, "
          f"acc {100 * op['accuracy']:.2f}%, P {100 * op['precision']:.2f}%, "
          f"R {100 * op['recall']:.2f}%, F1 {100 * op['f1']:.2f}%")

    E_t, y_t = extra["test"][0], extra["test"][1]
    groups = group_breakdown(E_t, y_t, ds_t.rows, cfg["modality"], min(n_imp, 200000), seed)
    if groups:
        print("  Разбивка test (галерея = эталон класса, пробы = группа):")
        for g, r in groups.items():
            print(f"    {g:28s} проб {r['n_probes']:5d} | EER {100 * r['eer']:5.2f}% | "
                  f"rank-1 {100 * r['rank1']:5.1f}%")

    timing = inference_time(model, INPUT_SHAPE[cfg["modality"]], device, cfg["timing_iters"], cfg["amp"])
    timing["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
    print("  Время на 1 изображение: " + ", ".join(f"{k} {v:.1f}" for k, v in timing.items()))

    plot_roc_det(curves, run_dir / "roc_det.png", title)
    plot_scores(gen_t, imp_t, thr, run_dir / "score_distribution.png", f"{title}: test")
    plot_groups(groups, run_dir / "groups_test.png", f"{title}: test по группам")

    metrics = {"run_name": cfg["run_name"], "modality": cfg["modality"], "backbone": cfg["backbone"],
               "fp_input": cfg["fp_input"] if cfg["modality"] == "fingerprint" else "",
               "arcface_s": cfg["arcface_s"], "arcface_m": cfg["arcface_m"], "lr": cfg["lr"],
               "best_epoch": int(ck["epoch"]) + 1,
               "epochs_trained": len(history) if history else None,
               "val": results["val"], "test": results["test"],
               "test_at_val_threshold": op, "test_groups": groups, "timing": timing,
               **git_info()}
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    flat = {f"final_{s}_{k}": v for s in ("val", "test") for k, v in results[s].items()
            if isinstance(v, float)}
    flat.update({f"final_test_op_{k}": v for k, v in op.items()})
    flat.update({f"final_{k}": v for k, v in timing.items()})
    for g, r in groups.items():
        key = g.replace("/", "_").replace("<", "lt").replace(">", "gt")
        flat[f"final_group_{key}_eer"] = r["eer"]
        flat[f"final_group_{key}_rank1"] = r["rank1"]
    tracker.metrics(flat)
    for f in ("metrics.json", "roc_det.png", "score_distribution.png", "groups_test.png",
              "best.pt", "train.log"):
        tracker.artifact(run_dir / f)
    print(f"\nРезультаты: {run_dir}")
    return metrics


# ======================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение мономодальной модели (ArcFace)")
    ap.add_argument("--config", help="YAML-конфиг (training/configs/*.yaml)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="переопределить параметры, напр. --set lr=5e-4 arcface_m=0.4")
    ap.add_argument("--resume", metavar="RUN_DIR", help="продолжить прогон с last.pt")
    ap.add_argument("--eval-only", metavar="RUN_DIR", help="только финальная оценка best.pt")
    args = ap.parse_args()
    config_path = str(Path(args.config).resolve()) if args.config else None
    existing = str(Path(args.resume or args.eval_only).resolve()) if (args.resume or args.eval_only) else None
    os.chdir(ROOT)

    if existing:
        run_dir = Path(existing)
        cfg = load_config(str(run_dir / "config.yaml"), args.set)
    else:
        cfg = load_config(config_path, args.set)
        if not cfg["run_name"]:
            short = {"mobilenetv3_large_100": "mobilenetv3"}.get(cfg["backbone"], cfg["backbone"])
            prefix = "fp" if cfg["modality"] == "fingerprint" else "iris"
            cfg["run_name"] = f"{prefix}_{short}_{datetime.now():%Y%m%d-%H%M}"
        run_dir = RUNS_DIR / cfg["run_name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    sys.stdout = Tee(run_dir / "train.log")
    print(f"\n##### {cfg['run_name']}  ({datetime.now():%Y-%m-%d %H:%M})  torch {torch.__version__}")
    seed_all(cfg["seed"])

    if args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_id = None
        if (run_dir / "last.pt").exists():
            run_id = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False).get("mlflow_run_id")
        tracker = Tracker(cfg, run_id)
        final_evaluation(cfg, run_dir, device, tracker)
        tracker.end()
    else:
        train(cfg, run_dir, resume=bool(args.resume))
    print("\nГотово")


if __name__ == "__main__":
    main()
