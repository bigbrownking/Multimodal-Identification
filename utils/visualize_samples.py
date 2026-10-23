import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fingerprint.preprocess_fingerprint import SUBSETS, parse_filename, imread_unicode
from iris.preprocess_iris import occlusion_mask

SEED = int(os.getenv("SEED", "42"))
OUT_DIR = Path(os.getenv("OUTPUT_ANALYSIS"))
SOCOFING_ROOT = Path(os.getenv("SOCOFING_ROOT"))
META_FP = Path(os.getenv("METADATA_SOCOFING") or os.getenv("METADATA_FINGERPRINT")
               or "data/processed/fingerprint_socofing_metadata.json")
META_IRIS = Path(os.getenv("METADATA_IRIS"))


def load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  Нет файла {path}")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ---------- отпечатки ----------
def socofing_sources() -> dict[tuple, Path]:
    out = {}
    for rel, name in SUBSETS:
        d = SOCOFING_ROOT / rel
        if not d.exists():
            continue
        for p in set(d.glob("*.BMP")) | set(d.glob("*.bmp")):
            info = parse_filename(p.stem)
            if info:
                out[(name, info["subject_id"], info["alteration"])] = p
    return out


def plot_fp_before_after(fp: list[dict], n_fingers: int = 3) -> None:
    if not fp:
        return
    rng = random.Random(SEED)
    sources = socofing_sources()
    by_class = defaultdict(dict)
    for r in fp:
        by_class[r["subject_id"]].setdefault(r["dataset"], []).append(r)
    levels = ["Real", "Altered-Easy", "Altered-Medium", "Altered-Hard"]
    full = [c for c, d in by_class.items() if all(l in d for l in levels)]
    chosen = rng.sample(sorted(full), min(n_fingers, len(full)))

    n_rows = 3
    fig, axes = plt.subplots(n_rows * len(chosen), len(levels),
                             figsize=(2.4 * len(levels), 2.6 * n_rows * len(chosen)))
    axes = np.atleast_2d(axes)
    for i, cls in enumerate(chosen):
        for j, lvl in enumerate(levels):
            r = rng.choice(by_class[cls][lvl])
            src = sources.get((lvl, cls, r["alteration"]))
            raw = imread_unicode(src) if src else None
            clahe = imread_unicode(Path(r["image_path_clahe"])) if r.get("image_path_clahe") else None
            proc = imread_unicode(Path(r["image_path"]))
            for k, (img, kind) in enumerate([(raw, "raw"), (clahe, "CLAHE"), (proc, "CLAHE+Gabor")]):
                ax = axes[n_rows * i + k, j]
                if img is not None:
                    ax.imshow(img, cmap="gray")
                ax.set_xticks([]); ax.set_yticks([])
                if k == 0:
                    ax.set_title(f"{lvl}\n{r['alteration']}", fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"{cls}\n{kind}", fontsize=8)
    fig.suptitle("Fingerprint: raw → CLAHE 224×224 → CLAHE + Gabor",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "fp_before_after.png")


# ---------- радужка ----------
def draw_segmentation(r: dict) -> np.ndarray | None:
    img = imread_unicode(Path(r["source_path"]))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if not r.get("pupil_found"):
        return rgb
    cx, cy, rp, ri = r["pupil_x"], r["pupil_y"], r["pupil_radius"], r["iris_radius"]
    ix, iy = r.get("iris_x", cx), r.get("iris_y", cy)
    _, occ = occlusion_mask(img, cx, cy, rp, ri, ix, iy)
    red = np.zeros_like(rgb); red[..., 0] = 255
    rgb[occ] = (0.55 * rgb[occ] + 0.45 * red[occ]).astype(np.uint8)
    t = max(1, img.shape[1] // 300)
    cv2.circle(rgb, (cx, cy), rp, (0, 220, 0), t)
    cv2.circle(rgb, (ix, iy), ri, (0, 140, 255), t)
    return rgb


def grid(records: list[dict], ncols: int, title: str, name: str) -> None:
    if not records:
        print(f"  {name}: нет примеров")
        return
    nrows = int(np.ceil(len(records) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, records):
        img = draw_segmentation(r)
        if img is not None:
            ax.imshow(img)
        ax.set_title(f"{r['subject_id']}\nocc={r['occlusion_ratio']:.2f}  q={r['quality_score']:.2f}",
                     fontsize=7)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, name)


def plot_iris_segmentation(iris: list[dict]) -> None:
    rng = random.Random(SEED)
    ok = [r for r in iris if r.get("pupil_found")]
    inter = [r for r in ok if r["dataset"] == "Interval"]
    thou = [r for r in ok if r["dataset"] == "Thousand"]
    sample = rng.sample(inter, min(6, len(inter))) + rng.sample(thou, min(18, len(thou)))
    grid(sample, 6, "Iris segmentation check (green: pupil, blue: iris, red: occlusion)",
         "iris_segmentation.png")


def plot_iris_occlusion_levels(iris: list[dict]) -> None:
    rng = random.Random(SEED + 1)
    ok = [r for r in iris if r.get("pupil_found")]
    rows = []
    for lo, hi in [(0.0, 0.15), (0.30, 0.40), (0.55, 1.01)]:
        pool = [r for r in ok if lo <= r["occlusion_ratio"] < hi]
        rows += rng.sample(pool, min(4, len(pool)))
    grid(rows, 4, "Occlusion levels: <0.15 (top), 0.30–0.40 (middle), ≥0.55 (bottom)",
         "iris_occlusion_levels.png")


def plot_iris_failures(iris: list[dict]) -> None:
    rng = random.Random(SEED + 2)
    bad = [r for r in iris if not r.get("pupil_found")]
    grid(rng.sample(bad, min(8, len(bad))), 4, f"Pupil not found ({len(bad)} images)",
         "iris_failures.png")


def plot_iris_before_after(iris: list[dict], n: int = 4) -> None:
    rng = random.Random(SEED + 3)
    ok = [r for r in iris if r.get("pupil_found")]
    sample = rng.sample(ok, min(n, len(ok)))
    if not sample:
        return
    fig, axes = plt.subplots(len(sample), 2, figsize=(11, 2.2 * len(sample)),
                             gridspec_kw={"width_ratios": [1, 2.6]})
    axes = np.atleast_2d(axes)
    for i, r in enumerate(sample):
        seg = draw_segmentation(r)
        strip = imread_unicode(Path(r["image_path"]))
        if seg is not None:
            axes[i, 0].imshow(seg)
        if strip is not None:
            axes[i, 1].imshow(strip, cmap="gray", aspect="auto")
        axes[i, 0].set_ylabel(r["subject_id"], fontsize=8)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_title("Raw + segmentation", fontsize=10)
    axes[0, 1].set_title("Rubber-sheet normalised strip 64×512 (CLAHE)", fontsize=10)
    fig.suptitle("Iris: raw vs processed", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "iris_before_after.png")


def run() -> None:
    fp = load(META_FP)
    iris = load(META_IRIS)
    if iris and "iris_x" not in iris[0]:
        raise SystemExit("Метаданные радужки старые — перезапустите шаг iris.")
    if fp and "image_path_clahe" not in fp[0]:
        raise SystemExit("Метаданные отпечатков старые — перезапустите шаг fingerprint.")
    plot_fp_before_after(fp)
    if iris:
        plot_iris_before_after(iris)
        plot_iris_segmentation(iris)
        plot_iris_occlusion_levels(iris)
        plot_iris_failures(iris)


if __name__ == "__main__":
    run()
