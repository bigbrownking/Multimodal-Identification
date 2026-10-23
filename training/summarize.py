import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"


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


def main() -> None:
    rows = []
    for f in sorted(RUNS_DIR.glob("*/metrics.json")) + sorted(RUNS_DIR.glob("*/eval_*/metrics.json")):
        with open(f, encoding="utf-8") as fh:
            rows.append(row(json.load(fh)))
    if not rows:
        print("Нет прогонов с metrics.json в", RUNS_DIR)
        return
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(RUNS_DIR / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (RUNS_DIR / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {RUNS_DIR / 'summary.csv'}, {RUNS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
