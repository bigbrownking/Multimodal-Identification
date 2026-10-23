import json
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from matplotlib.patches import Patch

load_dotenv()

OUT_DIR = Path(os.getenv("OUTPUT_ANALYSIS"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

FP_QUALITY_THRESH = float(os.getenv("FP_QUALITY_THRESH"))
IRIS_QUALITY_THRESH = float(os.getenv("IRIS_QUALITY_THRESH"))
SD302_MIN_FG = float(os.getenv("SD302_MIN_FG", "0.5"))
METADATA_SD302 = Path(os.getenv("METADATA_SD302", "data/processed/fingerprint_sd302_metadata.json"))

FP_COLOR = "#2E86AB"
SD302_COLOR = "#3BB273"
IRIS_COLOR = "#E84855"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sets(fp, iris, sd302):
    """(название, записи, цвет, поле качества, порог, подпись оси) — только непустые наборы."""
    out = [("Fingerprint (SOCOFing)", fp, FP_COLOR, "quality_score", FP_QUALITY_THRESH, "Quality Score"),
           ("Fingerprint (NIST SD 302)", sd302, SD302_COLOR, "fg_ratio", SD302_MIN_FG,
            "Finger share in 224×224 window"),
           ("Iris (CASIA-Interval + Thousand)", iris, IRIS_COLOR, "quality_score", IRIS_QUALITY_THRESH,
            "Quality Score")]
    return [s for s in out if s[1]]


def _save(fig, name: str) -> None:
    fig.tight_layout()
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def plot_quality(fp: list[dict], iris: list[dict], sd302: list[dict] | None = None) -> None:
    sets = _sets(fp, iris, sd302 or [])
    if not sets:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(6.5 * len(sets), 4), squeeze=False)
    fig.suptitle("Quality Score Distributions", fontsize=13, fontweight="bold")
    for ax, (label, meta, color, key, thresh, xlabel) in zip(axes[0], sets):
        scores = [float(r[key]) for r in meta if r.get(key, "") != ""]
        ax.hist(scores, bins=30, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(np.mean(scores), color="black", lw=1.8, ls="--", label=f"Mean = {np.mean(scores):.3f}")
        ax.axvline(thresh, color="red", lw=1.2, ls=":", label=f"Threshold = {thresh}")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    _save(fig, "quality_distributions.png")


def plot_lq_ratio(fp: list[dict], iris: list[dict], sd302: list[dict] | None = None) -> None:
    """Доля низкого качества: SOCOFing по уровням, SD 302 по устройствам, радужка по наборам."""
    labels, values, colors, counts = [], [], [], []

    def add(meta, key, prefix, color):
        groups: dict[str, list] = {}
        for r in meta:
            groups.setdefault(r.get(key, "?"), []).append(r)
        for g, recs in sorted(groups.items()):
            lq = sum(1 for r in recs if r["low_quality"])
            labels.append(f"{prefix} {g}")
            values.append(100 * lq / max(len(recs), 1))
            colors.append(color)
            counts.append(lq)

    add(fp, "dataset", "FP", FP_COLOR)
    add(sd302 or [], "device", "SD302", SD302_COLOR)
    add(iris, "dataset", "Iris", IRIS_COLOR)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(10, 1.05 * len(labels)), 4.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", alpha=0.85)
    ax.bar_label(bars, labels=[f"{v:.1f}%\n({n})" for v, n in zip(values, counts)], padding=3, fontsize=8)
    ax.set_ylabel("Low-Quality Samples (%)")
    ax.set_title("Proportion of Low-Quality Samples per Dataset", fontweight="bold")
    ax.set_ylim(0, max(values) * 1.35 if max(values) > 0 else 5)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    legend = [Patch(facecolor=FP_COLOR, label="Fingerprint (SOCOFing): quality score"),
              Patch(facecolor=IRIS_COLOR, label="Iris (CASIA): quality score")]
    if sd302:
        legend.insert(1, Patch(facecolor=SD302_COLOR, label=f"Fingerprint (SD 302): finger share < {SD302_MIN_FG}"))
    ax.legend(handles=legend, fontsize=8, loc="upper left")
    _save(fig, "low_quality_ratio.png")


def plot_occlusion(iris: list[dict]) -> None:
    if not iris:
        return
    occ = [r["occlusion_ratio"] for r in iris if not r["low_quality"]]
    if not occ:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(occ, bins=30, color=IRIS_COLOR, alpha=0.85, edgecolor="white")
    ax.axvline(np.mean(occ), color="black", lw=1.8, ls="--", label=f"Mean = {np.mean(occ):.3f}")
    ax.set_xlabel("Occlusion Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Iris Occlusion Ratio Distribution (Eyelids + Eyelashes)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "occlusion_distribution.png")


def plot_samples_per_class(fp: list[dict], iris: list[dict], sd302: list[dict] | None = None) -> None:
    sets = [(f"{name.split('(')[1].rstrip(')')}: samples per {'eye' if 'Iris' in name else 'finger'}",
             meta, color) for name, meta, color, *_ in _sets(fp, iris, sd302 or [])]
    if not sets:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(6.5 * len(sets), 4), squeeze=False)
    fig.suptitle("Samples per Class (good quality)", fontsize=13, fontweight="bold")
    for ax, (label, meta, color) in zip(axes[0], sets):
        counts: dict[str, int] = {}
        for r in meta:
            if not r.get("low_quality"):
                counts[r["subject_id"]] = counts.get(r["subject_id"], 0) + 1
        vals = np.array(list(counts.values()))
        bins = np.arange(vals.min(), vals.max() + 2) - 0.5
        ax.hist(vals, bins=bins, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(vals.mean(), color="black", lw=1.8, ls="--", label=f"Mean = {vals.mean():.2f}")
        ax.set_title(f"{label} ({len(vals)} classes)", fontsize=11)
        ax.set_xlabel("Samples per class")
        ax.set_ylabel("Classes")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    _save(fig, "samples_per_class.png")


def class_balance(meta: list[dict]) -> str:
    counts: dict[str, int] = {}
    for r in meta:
        if not r.get("low_quality"):
            counts[r["subject_id"]] = counts.get(r["subject_id"], 0) + 1
    if not counts:
        return "—"
    v = np.array(list(counts.values()))
    return f"{v.min()}/{np.median(v):.0f}/{v.max()}"


def print_summary(fp: list[dict], iris: list[dict], sd302: list[dict] | None = None) -> None:
    sd302 = sd302 or []
    cols = [("SOCOFing", fp), ("SD 302", sd302), ("Iris", iris)]
    cols = [(n, m) for n, m in cols if m]

    def mean(meta, key):
        vals = [float(r[key]) for r in meta if r.get(key, "") != ""]
        return f"{np.mean(vals):.3f}" if vals else "—"

    rows = [
        ("Total samples", lambda m: str(len(m))),
        ("Persons", lambda m: str(len({r["person_id"] for r in m}))),
        ("Classes (finger / eye)", lambda m: str(len({r["subject_id"] for r in m}))),
        ("Samples/class min/med/max", class_balance),
        ("Mean quality score", lambda m: mean(m, "quality_score")),
        ("Mean finger share in window", lambda m: mean(m, "fg_ratio")),
        ("Median ridge period, px", lambda m: (f"{np.median([float(r['ridge_period']) for r in m if r.get('ridge_period', '') != '']):.1f}"
                                               if any(r.get("ridge_period", "") != "" for r in m) else "—")),
        ("Mean occlusion ratio", lambda m: mean(m, "occlusion_ratio")),
        ("Pupil not found", lambda m: str(sum(1 for r in m if not r.get("pupil_found", True)))
                                      if m and "occlusion_ratio" in m[0] else "—"),
        ("Low-quality count", lambda m: str(sum(1 for r in m if r.get("low_quality")))),
    ]
    width = 35 + 12 * len(cols)
    print("\n" + "=" * width)
    print(f"{'Metric':<35}" + "".join(f"{n:>12}" for n, _ in cols))
    print("=" * width)
    for label, fn in rows:
        print(f"  {label:<33}" + "".join(f"{fn(m):>12}" for _, m in cols))
    print("=" * width)


def run() -> None:
    fp = load(Path(os.getenv("METADATA_SOCOFING") or os.getenv("METADATA_FINGERPRINT")
                   or "data/processed/fingerprint_socofing_metadata.json"))
    iris = load(Path(os.getenv("METADATA_IRIS")))
    sd302 = load(METADATA_SD302)
    print_summary(fp, iris, sd302)
    print("\nGenerating plots...")
    plot_quality(fp, iris, sd302)
    plot_lq_ratio(fp, iris, sd302)
    plot_occlusion(iris)
    plot_samples_per_class(fp, iris, sd302)
    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    run()
