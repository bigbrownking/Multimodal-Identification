"""
Предобработка отпечатков пальцев: два набора в одном файле.

  SOCOFing      — 600 человек × 10 пальцев, один реальный снимок + синтетические искажения
                  (Real / Altered-Easy / Medium / Hard).
  NIST SD 302   — плоские снимки одного пальца с нескольких сенсоров = настоящие повторные
                  прикладывания (нужен SD302_ROOT в .env; если не задан — набор пропускается).

  python run.py --step fingerprint          # оба набора
  python run.py --step sd302                # только SD 302
  python fingerprint/preprocess_fingerprint.py [socofing|sd302|all]
"""
import cv2
import numpy as np
from pathlib import Path
import json
import os
import re
from collections import Counter, defaultdict
from dotenv import load_dotenv

load_dotenv()

DATASET_ROOT = Path(os.getenv("SOCOFING_ROOT"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_SOCOFING") or os.getenv("OUTPUT_FINGERPRINT")
                  or "data/processed/fingerprint_socofing")
OUTPUT_DIR_CLAHE = Path(os.getenv("OUTPUT_SOCOFING_CLAHE") or os.getenv("OUTPUT_FINGERPRINT_CLAHE")
                        or str(OUTPUT_DIR) + "_clahe")
META_PATH = Path(os.getenv("METADATA_SOCOFING") or os.getenv("METADATA_FINGERPRINT")
                 or "data/processed/fingerprint_socofing_metadata.json")
TARGET_SIZE = (224, 224)
QUALITY_THRESH = float(os.getenv("FP_QUALITY_THRESH"))
GABOR_LAMBDA = 10.0
GABOR_SIGMA = 4.0
GABOR_KSIZE = 21


SUBSETS = [
    ("Real",                       "Real"),
    ("Altered/Altered-Easy",       "Altered-Easy"),
    ("Altered/Altered-Medium",     "Altered-Medium"),
    ("Altered/Altered-Hard",       "Altered-Hard"),
]


def imread_unicode(path: Path) -> np.ndarray | None:
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        buf.tofile(str(path))


def crop_border(img: np.ndarray, max_trim: int = 12) -> np.ndarray:
    h, w = img.shape

    def dark(line: np.ndarray) -> bool:
        return line.mean() < 40

    def gray_line(line: np.ndarray) -> bool:
        lo, hi = np.percentile(line, [10, 90])
        return 40 <= line.mean() <= 215 and hi - lo < 25

    def depth(get_line, n: int) -> int:
        k = 0
        while k < min(max_trim, n - 1) and dark(get_line(k)):
            k += 1
        for j in range(k, min(k + 3, n - 1)):
            if gray_line(get_line(j)):
                k = j + 1
        return k

    b = h - depth(lambda k: img[h - 1 - k, :], h)
    r = w - depth(lambda k: img[:b, w - 1 - k], w)
    t = depth(lambda k: img[k, :r], b)
    l = depth(lambda k: img[t:b, k], r)
    return img[t:b, l:r]


def gabor_enhance(img: np.ndarray) -> np.ndarray:
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)
    responses = []
    for theta in np.arange(0, np.pi, np.pi / 8):
        kernel = cv2.getGaborKernel(
            (GABOR_KSIZE, GABOR_KSIZE), sigma=GABOR_SIGMA, theta=theta,
            lambd=GABOR_LAMBDA, gamma=0.5, psi=0, ktype=cv2.CV_32F
        )
        kernel -= kernel.mean()
        responses.append(cv2.filter2D(img, cv2.CV_32F, kernel))
    enhanced = np.max(np.stack(responses), axis=0)
    return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def load_and_enhance(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = imread_unicode(path)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть: {path}")

    img = crop_border(img)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    base = clahe.apply(img)
    return base, gabor_enhance(base)


def quality_score(img: np.ndarray) -> float:
    lap       = cv2.Laplacian(img, cv2.CV_64F).var()
    sharpness = min(lap / 2000.0, 1.0)
    contrast  = img.std() / 128.0
    return float(np.clip(0.6 * sharpness + 0.4 * contrast, 0.0, 1.0))


FNAME_RE = re.compile(
    r"^(?P<person>\d+)__(?P<gender>[MF])_(?P<hand>Left|Right)_"
    r"(?P<finger>thumb|index|middle|ring|little)_finger"
    r"(?:_(?P<alt>CR|Obl|Zcut))?$",
    re.IGNORECASE,
)


def parse_filename(stem: str) -> dict | None:
    m = FNAME_RE.match(stem)
    if m is None:
        return None
    person = m["person"]
    hand = m["hand"].capitalize()
    finger = m["finger"].lower()
    alt = m["alt"] or "none"
    return {
        "person_id":  person,
        "subject_id": f"{person}_{hand}_{finger}",
        "hand":       hand,
        "finger":     finger,
        "alteration": alt,
    }


def process_subset(subset_path: Path, subset_name: str) -> list[dict]:
    bmp_files = sorted(subset_path.glob("*.BMP")) + sorted(subset_path.glob("*.bmp"))
    bmp_files = sorted(set(bmp_files))
    print(f"\n[{subset_name}] Найдено файлов: {len(bmp_files)}")

    all_meta = []
    processed = 0
    skipped = 0

    for bmp_path in bmp_files:
        info = parse_filename(bmp_path.stem)
        if info is None:
            skipped += 1
            continue

        sample_id = f"{subset_name}_{info['alteration']}"
        fname = f"{info['hand']}_{info['finger']}_{info['alteration']}.png"
        out_path = OUTPUT_DIR / subset_name / info["person_id"] / fname
        out_path_clahe = OUTPUT_DIR_CLAHE / subset_name / info["person_id"] / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path_clahe.parent.mkdir(parents=True, exist_ok=True)

        raw = imread_unicode(bmp_path)
        if raw is None:
            continue
        qs = quality_score(raw)

        try:
            base, enhanced = load_and_enhance(bmp_path)
            imwrite_unicode(out_path, enhanced)
            imwrite_unicode(out_path_clahe, base)
        except Exception as e:
            print(f"  Ошибка: {bmp_path.name}: {e}")
            continue

        all_meta.append({
            "dataset":       subset_name,
            "person_id":     info["person_id"],
            "subject_id":    info["subject_id"],
            "sample_id":     sample_id,
            "hand":          info["hand"],
            "finger":        info["finger"],
            "alteration":    info["alteration"],
            "image_path":    str(out_path),
            "image_path_clahe": str(out_path_clahe),
            "quality_score": round(qs, 4),
            "low_quality":   qs < QUALITY_THRESH,
        })
        processed += 1
        if processed % 500 == 0:
            print(f"  {processed}/{len(bmp_files)} обработано...")

    lq = sum(1 for r in all_meta if r["low_quality"])
    print(f"  [{subset_name}] Готово: {processed} изображений, "
          f"низкое качество: {lq} ({100*lq/max(processed,1):.1f}%)")
    if skipped:
        print(f"  [{subset_name}] Пропущено файлов с нестандартным именем: {skipped}")
    return all_meta


def process_socofing() -> list[dict]:
    all_meta = []
    for rel_path, subset_name in SUBSETS:
        subset_path = DATASET_ROOT / rel_path
        if not subset_path.exists():
            print(f"  [{subset_name}] Папка не найдена: {subset_path} — пропуск")
            continue
        all_meta += process_subset(subset_path, subset_name)
    return all_meta


def save_metadata(meta: list[dict], path: Path = META_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    lq = sum(1 for r in meta if r["low_quality"])
    persons = len({r["person_id"] for r in meta})
    per_cls = Counter(r["subject_id"] for r in meta)
    n = np.array(list(per_cls.values()))
    print(f"\nМетаданные сохранены → {path}")
    print(f"Итого: {len(meta)} образцов, {persons} человек, {len(per_cls)} классов (пальцев), "
          f"снимков на палец min/med/max = {n.min()}/{int(np.median(n))}/{n.max()}, "
          f"низкое качество: {lq} ({100*lq/max(len(meta),1):.1f}%)")


# ======================================================================
# NIST SD 302
# ======================================================================
# Источники: images_d — plain с устройств K, L, M, P; images_b — отдельные пальцы, вырезанные
# из слэпов 4-4-2 (тип slap, FRGP 01–10). Прокатанные (roll) по умолчанию не берём: другая форма
# отпечатка; включить — SD302_CAPTURES=plain,slap,roll. Целые слэпы (FRGP 13–15) пропускаются.
# Масштаб: снимок приводится к 500 ppi и уменьшается в SD302_SCALE раз, чтобы период гребней
# совпал с SOCOFing; затем вырезается окно 224×224 вокруг центра пальца. Период проверяется
# по спектру и печатается в логе.

SD302_ROOT_ENV = os.getenv("SD302_ROOT", "").strip()
SD302_ROOT = Path(SD302_ROOT_ENV) if SD302_ROOT_ENV else None
SD302_OUTPUT_DIR = Path(os.getenv("OUTPUT_FINGERPRINT_SD302", "data/processed/fingerprint_sd302"))
SD302_META_PATH = Path(os.getenv("METADATA_SD302", "data/processed/fingerprint_sd302_metadata.json"))
OUTPUT_ANALYSIS = Path(os.getenv("OUTPUT_ANALYSIS", "data/analysis_plots"))
SD302_CAPTURES = {c.strip() for c in os.getenv("SD302_CAPTURES", "plain,slap").split(",") if c.strip()}
SD302_MIN_FG = float(os.getenv("SD302_MIN_FG", "0.5"))
SD302_SEED = int(os.getenv("SEED", "42"))

SD302_PPI = 500
SD302_SCALE = float(os.getenv("SD302_SCALE", "0.77"))
SD302_FINGERS = {1: ("Right", "thumb"), 2: ("Right", "index"), 3: ("Right", "middle"),
           4: ("Right", "ring"), 5: ("Right", "little"), 6: ("Left", "thumb"),
           7: ("Left", "index"), 8: ("Left", "middle"), 9: ("Left", "ring"), 10: ("Left", "little")}

SD302_RE = re.compile(r"^(?P<subj>\d+)_(?P<dev>[A-Za-z0-9]+)_(?P<ppi>\d+)_"
                      r"(?P<cap>[a-z]+)_(?P<frgp>\d{2})$")


# ---------- поиск файлов ----------
def sd302_scan(root: Path) -> tuple[list[dict], Counter]:
    best: dict[tuple, dict] = {}
    skipped = Counter()
    for p in root.rglob("*.png"):
        m = SD302_RE.match(p.stem)
        if m is None:
            skipped["имя не по шаблону"] += 1
            continue
        frgp, cap, ppi = int(m["frgp"]), m["cap"], int(m["ppi"])
        if frgp not in SD302_FINGERS:
            skipped[f"FRGP {frgp:02d} (не один палец)"] += 1
            continue
        if cap not in SD302_CAPTURES:
            skipped[f"тип {cap}"] += 1
            continue
        key = (m["subj"], m["dev"], cap, frgp)
        rec = {"path": p, "subj": m["subj"], "dev": m["dev"], "cap": cap, "frgp": frgp, "ppi": ppi}
        old = best.get(key)
        if old is None or abs(ppi - SD302_PPI) < abs(old["ppi"] - SD302_PPI):
            if old is not None:
                skipped["дубликат в другом разрешении"] += 1
            best[key] = rec
        else:
            skipped["дубликат в другом разрешении"] += 1
    return sorted(best.values(), key=lambda r: (r["subj"], r["frgp"], r["dev"], r["cap"])), skipped


# ---------- сегментация и вырезание окна ----------
def foreground_mask(img: np.ndarray, blk: int = 16) -> np.ndarray:
    f = img.astype(np.float32)
    mean = cv2.blur(f, (blk, blk))
    std = np.sqrt(np.maximum(cv2.blur(f * f, (blk, blk)) - mean * mean, 0))
    std8 = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(std8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros_like(mask, bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def crop_window(img: np.ndarray, mask: np.ndarray, size: int = TARGET_SIZE[0]):
    bg = int(np.median(img[~mask])) if (~mask).any() else 255
    h, w = img.shape
    rows = np.nonzero(mask.mean(1) > 0.15)[0] if mask.any() else np.array([], int)
    if len(rows):
        top = int(rows[0])
        band = mask[top:top + size]
        xs = np.nonzero(band)[1]
        cx = int(xs.mean()) if len(xs) else w // 2
        cy = max(top - 8, 0) + size // 2
    else:
        cy, cx = h // 2, w // 2
    pad = size
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg)
    bigm = cv2.copyMakeBorder(mask.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    y0 = int(np.clip(cy - size // 2, 0, max(h - size, 0))) + pad if h >= size else cy - size // 2 + pad
    x0 = int(np.clip(cx - size // 2, 0, max(w - size, 0))) + pad if w >= size else cx - size // 2 + pad
    win = big[y0:y0 + size, x0:x0 + size]
    winm = bigm[y0:y0 + size, x0:x0 + size].astype(bool)
    return win, winm, (x0 - pad, y0 - pad)


def finger_share(winm: np.ndarray, origin: tuple[int, int], img_shape: tuple[int, int]) -> float:
    x0, y0 = origin
    h, w = img_shape
    size = winm.shape[0]
    ys, ye = max(0, -y0), min(size, h - y0)
    xs, xe = max(0, -x0), min(size, w - x0)
    if ye <= ys or xe <= xs:
        return 0.0
    return float(winm[ys:ye, xs:xe].mean())


def ridge_period(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    h, w = img.shape
    c = img[h // 2 - 64:h // 2 + 64, w // 2 - 64:w // 2 + 64].astype(np.float32)
    if mask is not None and mask[h // 2 - 64:h // 2 + 64, w // 2 - 64:w // 2 + 64].mean() < 0.7:
        return float("nan")
    c = (c - c.mean()) * np.outer(np.hanning(128), np.hanning(128))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(c)))
    yy, xx = np.indices(spec.shape)
    r = np.hypot(yy - 64, xx - 64).astype(int)
    prof = np.bincount(r.ravel(), spec.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    lo, hi = int(128 / 20), int(128 / 4)
    k = lo + int(np.argmax(prof[lo:hi + 1]))
    return 128.0 / k


def sd302_process_image(raw: np.ndarray, ppi: int):
    s = SD302_PPI / ppi * SD302_SCALE
    if abs(s - 1.0) > 1e-3:
        raw = cv2.resize(raw, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mask = foreground_mask(raw)
    win, winm, origin = crop_window(raw, mask)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(win)
    return raw, mask, clahe, winm, origin


# ---------- основной проход ----------
def process_sd302(root: Path | None = SD302_ROOT) -> list[dict]:
    if root is None:
        print("  SD302_ROOT не задан в .env — набор SD 302 пропущен")
        return []
    if not root.exists():
        print(f"  SD302_ROOT не найден: {root} — проверьте путь в .env")
        return []
    files, skipped = sd302_scan(root)
    print(f"  Найдено снимков одного пальца: {len(files)} (типы: {sorted(SD302_CAPTURES)})")
    for k, v in skipped.most_common():
        print(f"    пропущено — {k}: {v}")
    by = Counter((r["dev"], r["cap"], r["ppi"]) for r in files)
    for (dev, cap, ppi), n in sorted(by.items()):
        print(f"    устройство {dev:3s} {cap:9s} {ppi:4d} ppi: {n} снимков")

    meta, periods, failed = [], [], 0
    for i, r in enumerate(files):
        raw = imread_unicode(r["path"])
        if raw is None:
            failed += 1
            continue
        scaled, _, out, winm, origin = sd302_process_image(raw, r["ppi"])
        fg = finger_share(winm, origin, scaled.shape)
        per = ridge_period(out, winm)
        if np.isfinite(per):
            periods.append(per)
        hand, finger = SD302_FINGERS[r["frgp"]]
        out_path = SD302_OUTPUT_DIR / r["subj"] / f"{r['frgp']:02d}_{r['dev']}_{r['cap']}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(out_path, out)
        qs = quality_score(out)
        meta.append({
            "dataset": "SD302",
            "person_id": f"SD302_{r['subj']}",
            "subject_id": f"SD302_{r['subj']}_{r['frgp']:02d}",
            "sample_id": f"{r['dev']}_{r['cap']}",
            "device": r["dev"], "capture": r["cap"], "source_ppi": r["ppi"],
            "hand": hand, "finger": finger, "alteration": "none",
            "image_path": str(out_path), "image_path_clahe": str(out_path),
            "quality_score": round(qs, 4),
            "fg_ratio": round(fg, 4),
            "ridge_period": round(per, 2) if np.isfinite(per) else "",
            "low_quality": fg < SD302_MIN_FG,
        })
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(files)} обработано...")

    lq = sum(r["low_quality"] for r in meta)
    print(f"  Готово: {len(meta)} снимков, не открылись: {failed}, "
          f"мало пальца в окне (<{SD302_MIN_FG:.0%}): {lq} ({100 * lq / max(len(meta), 1):.1f}%)")
    sd302_report_period(periods)
    return meta


def sd302_report_period(periods: list[float]) -> None:
    if periods:
        print(f"  Период гребней SD302 (px): медиана {np.median(periods):.1f}, "
              f"IQR {np.percentile(periods, 25):.1f}–{np.percentile(periods, 75):.1f}")
    soc = sorted((OUTPUT_DIR_CLAHE / "Real").rglob("*.png"))[:300] if (OUTPUT_DIR_CLAHE / "Real").exists() else []
    sp = [ridge_period(im) for im in (imread_unicode(p) for p in soc) if im is not None]
    sp = [p for p in sp if np.isfinite(p)]
    if sp:
        print(f"  Период гребней SOCOFing после 224×224 (px): медиана {np.median(sp):.1f} "
              f"— масштабы должны быть близки")


def plot_sd302_before_after(meta: list[dict], root: Path | None = SD302_ROOT, n: int = 6) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not meta or root is None:
        return
    files, _ = sd302_scan(root)
    by_key = {(f["subj"], f["dev"], f["cap"], f["frgp"]): f for f in files}
    rng = np.random.default_rng(SD302_SEED)
    cls = defaultdict(list)
    for r in meta:
        cls[r["subject_id"]].append(r)
    picks = [cls[k] for k in rng.choice(sorted(cls), size=min(n // 2, len(cls)), replace=False)]
    rows = []
    for recs in picks:
        for r in recs[:2]:
            subj = r["person_id"].split("_", 1)[1]
            f = by_key.get((subj, r["device"], r["capture"], int(r["subject_id"][-2:])))
            if f is None:
                continue
            raw0 = imread_unicode(f["path"])
            raw, mask, out, _, (x0, y0) = sd302_process_image(raw0, f["ppi"])
            vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnt, -1, (0, 200, 0), 2)
            cv2.rectangle(vis, (x0, y0), (x0 + TARGET_SIZE[0], y0 + TARGET_SIZE[0]), (0, 0, 255), 2)
            rows.append((vis[..., ::-1], out, f"{r['subject_id']} · {r['device']} {r['capture']}"))
    if not rows:
        return
    fig, axes = plt.subplots(2, len(rows), figsize=(2.6 * len(rows), 6))
    axes = np.array(axes).reshape(2, -1)
    for j, (vis, out, title) in enumerate(rows):
        axes[0, j].imshow(vis)
        axes[0, j].set_title(title, fontsize=7)
        axes[1, j].imshow(out, cmap="gray")
        for a in axes[:, j]:
            a.set_xticks([])
            a.set_yticks([])
    axes[0, 0].set_ylabel("исходный (зелёный — палец,\nкрасный — окно 224×224)", fontsize=8)
    axes[1, 0].set_ylabel("после CLAHE", fontsize=8)
    fig.suptitle("NIST SD 302: one finger, different devices", fontweight="bold")
    fig.tight_layout()
    OUTPUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
    p = OUTPUT_ANALYSIS / "sd302_before_after.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def run_socofing() -> list[dict]:
    meta = process_socofing()
    if meta:
        save_metadata(meta, META_PATH)
    else:
        print("Изображения SOCOFing не найдены. Проверьте SOCOFING_ROOT.")
    return meta


def run_sd302() -> list[dict]:
    meta = process_sd302()
    if meta:
        save_metadata(meta, SD302_META_PATH)
        plot_sd302_before_after(meta)
    return meta


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("socofing", "all"):
        print("=" * 55 + "\nПредобработка отпечатков: SOCOFing\n" + "=" * 55)
        run_socofing()
    if which in ("sd302", "all"):
        print("=" * 55 + "\nПредобработка отпечатков: NIST SD 302\n" + "=" * 55)
        run_sd302()