"""
Сводные таблицы всех прогонов для отчёта.

  python training/summarize.py
    -> runs/summary.md, runs/summary.csv                — мономодальные модели (train.py)
    -> runs/summary_fusion.md, runs/summary_fusion.csv  — слияние (fusion.py, папки runs/fusion_*)
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
FUSION_METHODS = ("fp_only", "iris_only", "score_sum", "score_weighted", "feat_concat", "feat_learned")


def pct(x):
    return "" if x is None else f"{100 * x:.2f}"


def row(m: dict) -> dict:
    v, t, op, tm = m["val"], m["test"], m["test_at_val_threshold"], m["timing"]
    r = {
        "run": m["run_name"], "modality": m["modality"], "backbone": m["backbone"],
        "fp_input": m.get("fp_input", ""), "s": m["arcface_s"], "m": m["arcface_m"], "lr": m["lr"],
        "best_epoch": m["best_epoch"],
        "val_EER_%": pct(v["eer"]), "test_EER_%": pct(t["eer"]), "test_AUC": f"{t['auc']:.4f}",
        "test_TAR@FAR=0.1%": pct(t.get("tar_at_far_1e-3")),
        "test_FAR_%": pct(op["far"]), "test_FRR_%": pct(op["frr"]),
        "acc_%": pct(op["accuracy"]), "prec_%": pct(op["precision"]),
        "rec_%": pct(op["recall"]), "F1_%": pct(op["f1"]),
        "rank1_%": pct(t.get("gp_rank1")), "rank5_%": pct(t.get("gp_rank5")),
        "gpu_ms": f"{tm.get('gpu_ms_per_image', float('nan')):.1f}",
        "cpu_ms": f"{tm['cpu_ms_per_image']:.1f}", "params_M": f"{tm['params_m']:.1f}",
    }
    for g, gr in m.get("test_groups", {}).items():
        r[f"EER_{g}_%"] = pct(gr["eer"])
    return r


def fusion_row(name: str, m: dict) -> dict:
    res = m["results"]
    r = {"run": name, "corpus": m["corpus"], "fp_run": m["fp_run"], "iris_run": m["iris_run"],
         "w*": f"{m['w_best']:.2f}"}
    for meth in FUSION_METHODS:
        if meth in res:
            r[f"{meth}_val_EER_%"] = pct(res[meth]["val"]["eer"])
            r[f"{meth}_test_EER_%"] = pct(res[meth]["test"]["eer"])
    best = min((meth for meth in FUSION_METHODS if meth in res and meth not in ("fp_only", "iris_only")),
               key=lambda k: res[k]["val"]["eer"])
    bt = res[best]["test"]
    r.update({"best_by_val": best, "best_test_EER_%": pct(bt["eer"]),
              "best_TAR@FAR=0.1%": pct(bt.get("tar_at_far_1e-3")), "best_F1_%": pct(bt.get("op_f1")),
              "best_rank1_%": pct(bt.get("gp_rank1")),
              "cpu_ms_pair": f"{m.get('timing_ms', {}).get('cpu', float('nan')):.1f}",
              "gpu_ms_pair": f"{m.get('timing_ms', {}).get('gpu', float('nan')):.1f}"})
    return r


def write(rows: list[dict], stem: str, title: str) -> None:
    if not rows:
        return
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(RUNS_DIR / f"{stem}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (RUNS_DIR / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n### {title}\n" + "\n".join(md))
    print(f"-> {RUNS_DIR / (stem + '.md')}, {RUNS_DIR / (stem + '.csv')}")


def main() -> None:
    mono, fusion, skipped = [], [], []
    files = sorted(RUNS_DIR.glob("*/metrics.json")) + sorted(RUNS_DIR.glob("*/eval_*/metrics.json"))
    for f in files:
        with open(f, encoding="utf-8") as fh:
            m = json.load(fh)
        if "results" in m and "corpus" in m:
            fusion.append(fusion_row(f.parent.name, m))
        elif "val" in m and "test" in m:
            mono.append(row(m))
        else:
            skipped.append(str(f.relative_to(ROOT)))
    if not mono and not fusion:
        print("Нет прогонов с metrics.json в", RUNS_DIR)
        return
    write(mono, "summary", "Мономодальные модели")
    write(fusion, "summary_fusion", "Слияние (лучший метод выбран по val EER)")
    if skipped:
        print("\nПропущены (неизвестный формат):", ", ".join(skipped))


if __name__ == "__main__":
    main()