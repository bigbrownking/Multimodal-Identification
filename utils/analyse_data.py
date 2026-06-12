import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv

load_dotenv()

OUT_DIR = Path(os.getenv("OUTPUT_ANALYSIS"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

FP_QUALITY_THRESH = float(os.getenv("FP_QUALITY_THRESH"))
IRIS_QUALITY_THRESH = float(os.getenv("IRIS_QUALITY_THRESH"))
FP_COLOR   = "#2E86AB"
IRIS_COLOR = "#E84855"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_quality(fp: list[dict], iris: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Quality Score Distributions", fontsize=13, fontweight="bold")

    for ax, meta, label, color, thresh in [
        (axes[0], fp, "Fingerprint (SOCOFing)", FP_COLOR, FP_QUALITY_THRESH),
        (axes[1], iris, "Iris (CASIA-Interval + Thousand)", IRIS_COLOR, IRIS_QUALITY_THRESH),
    ]:
        if not meta:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(label)
            continue
        scores = [r["quality_score"] for r in meta]
        ax.hist(scores, bins=30, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(np.mean(scores), color="black", lw=1.8, ls="--",
                   label=f"Mean = {np.mean(scores):.3f}")
        ax.axvline(thresh, color="red", lw=1.2, ls=":",
                   label=f"Threshold = {thresh}")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Quality Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = OUT_DIR / "quality_distributions.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def plot_lq_ratio(fp: list[dict], iris: list[dict]) -> None:
    labels, values, colors = [], [], []

    fp_by_db: dict[str, list] = {}
    for r in fp:
        fp_by_db.setdefault(r.get("dataset", "?"), []).append(r)
    for db, recs in sorted(fp_by_db.items()):
        lq = sum(1 for r in recs if r["low_quality"])
        labels.append(f"FP {db}")
        values.append(100 * lq / max(len(recs), 1))
        colors.append(FP_COLOR)

    iris_by_ds: dict[str, list] = {}
    for r in iris:
        iris_by_ds.setdefault(r.get("dataset", "?"), []).append(r)
    for ds, recs in sorted(iris_by_ds.items()):
        lq = sum(1 for r in recs if r["low_quality"])
        labels.append(f"Iris {ds}")
        values.append(100 * lq / max(len(recs), 1))
        colors.append(IRIS_COLOR)

    if not labels:
        return

    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", alpha=0.85)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylabel("Low-Quality Samples (%)")
    ax.set_title("Proportion of Low-Quality Samples per Dataset", fontweight="bold")
    ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 5)
    ax.grid(axis="y", alpha=0.3)
    legend_elements = [Patch(facecolor=FP_COLOR,   label="Fingerprint (SOCOFing)"),
                       Patch(facecolor=IRIS_COLOR, label="Iris (CASIA)")]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")
    plt.tight_layout()
    p = OUT_DIR / "low_quality_ratio.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def plot_occlusion(iris: list[dict]) -> None:
    if not iris:
        return
    occ = [r["occlusion_ratio"] for r in iris if not r["low_quality"]]
    if not occ:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(occ, bins=30, color=IRIS_COLOR, alpha=0.85, edgecolor="white")
    ax.axvline(np.mean(occ), color="black", lw=1.8, ls="--",
               label=f"Mean = {np.mean(occ):.3f}")
    ax.set_xlabel("Occlusion Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Iris Occlusion Ratio Distribution (Eyelids + Eyelashes)",
                 fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = OUT_DIR / "occlusion_distribution.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def print_summary(fp: list[dict], iris: list[dict]) -> None:
    print("\n" + "=" * 60)
    print(f"{'Metric':<35} {'Fingerprint':>12} {'Iris':>10}")
    print("=" * 60)

    def s(meta, key):
        if not meta:
            return "—"
        vals = [r[key] for r in meta if key in r]
        return f"{np.mean(vals):.3f}" if vals else "—"

    rows = [
        ("Total samples",          str(len(fp)),  str(len(iris))),
        ("Unique subjects",
         str(len({r["subject_id"] for r in fp}))   if fp   else "—",
         str(len({r["subject_id"] for r in iris})) if iris else "—"),
        ("Mean quality score",     s(fp, "quality_score"), s(iris, "quality_score")),
        ("Low-quality count",
         str(sum(1 for r in fp   if r.get("low_quality"))),
         str(sum(1 for r in iris if r.get("low_quality")))),
    ]
    for label, v1, v2 in rows:
        print(f"  {label:<33} {v1:>12} {v2:>10}")
    print("=" * 60)


if __name__ == "__main__":
    fp_meta   = load(Path(os.getenv("METADATA_FINGERPRINT")))
    iris_meta = load(Path(os.getenv("METADATA_IRIS")))

    print_summary(fp_meta, iris_meta)
    print("\nGenerating plots...")
    plot_quality(fp_meta, iris_meta)
    plot_lq_ratio(fp_meta, iris_meta)
    plot_occlusion(iris_meta)
    print(f"\nAll plots saved to {OUT_DIR}/")