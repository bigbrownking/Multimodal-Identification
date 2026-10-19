import cv2
import numpy as np
from pathlib import Path
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

DATASET_ROOT = Path(os.getenv("SOCOFING_ROOT"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_FINGERPRINT"))
OUTPUT_DIR_CLAHE = Path(os.getenv("OUTPUT_FINGERPRINT_CLAHE", str(OUTPUT_DIR) + "_clahe"))
META_PATH = Path(os.getenv("METADATA_FINGERPRINT"))
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
        for j in range(k, min(k + 3, n - 1)):   # линия рамки в 1–3 px от края
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
        kernel -= kernel.mean()          # нулевое среднее: фон не даёт отклика
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


def save_metadata(meta: list[dict]) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    lq = sum(1 for r in meta if r["low_quality"])
    persons = len({r["person_id"] for r in meta})
    classes = len({r["subject_id"] for r in meta})
    print(f"\nМетаданные сохранены → {META_PATH}")
    print(f"Итого: {len(meta)} образцов, {persons} человек, {classes} классов (пальцев), "
          f"низкое качество: {lq} ({100*lq/max(len(meta),1):.1f}%)")


if __name__ == "__main__":
    print("=" * 55)
    print("Предобработка отпечатков пальцев (SOCOFing)")
    print("=" * 55)
    meta = process_socofing()
    if meta:
        save_metadata(meta)
    else:
        print("Изображения не найдены. Проверьте DATASET_ROOT.")