import cv2
import numpy as np
from pathlib import Path
import json
import os
from dotenv import load_dotenv

load_dotenv()

INTERVAL_ROOT = Path(os.getenv("CASIA_INTERVAL_ROOT"))
THOUSAND_ROOT = Path(os.getenv("CASIA_THOUSAND_ROOT"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_IRIS"))
META_PATH = Path(os.getenv("METADATA_IRIS"))
RUBBER_H = 64
RUBBER_W = 512
QUALITY_THRESH = float(os.getenv("IRIS_QUALITY_THRESH"))

OCC_K = 2.5
OCC_MIN_SIGMA = 4.0
SPECULAR_LEVEL = 240
REF_ANGLE_DEG = 35

def imread_unicode(path: Path) -> np.ndarray | None:
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        buf.tofile(str(path))


def find_pupil(img: np.ndarray):
    blurred = cv2.GaussianBlur(img, (7, 7), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.0,
        minDist=30,
        param1=100, param2=15,
        minRadius=15,
        maxRadius=75,
    )
    if circles is None:
        return None
    c = np.round(circles[0, 0]).astype(int)
    return int(c[0]), int(c[1]), int(c[2])


def find_iris(img: np.ndarray, cx: int, cy: int, r_p: int) -> tuple[int, bool]:
    h, w = img.shape
    masked = img.copy()
    cv2.circle(masked, (cx, cy), r_p + 5, int(img.mean()), -1)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(masked, (5, 5), 1),
        cv2.HOUGH_GRADIENT, dp=1.0,
        minDist=30,
        param1=80, param2=12,
        minRadius=int(r_p * 1.5),
        maxRadius=min(140, w // 2),
    )
    if circles is None:
        return min(int(r_p * 2.5), 130), False
    c = np.round(circles[0, 0]).astype(int)
    return int(c[2]), True


def rubber_sheet(img: np.ndarray, cx: int, cy: int,
                 r_p: int, r_i: int) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, RUBBER_W, endpoint=False)
    rho   = np.linspace(0, 1, RUBBER_H)
    R = r_p + rho[:, None] * (r_i - r_p)
    X = (cx + R * np.cos(theta[None, :])).astype(np.float32)
    Y = (cy - R * np.sin(theta[None, :])).astype(np.float32)
    strip = cv2.remap(img, X, Y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 16))
    return clahe.apply(strip)


def occlusion_mask(img: np.ndarray, cx: int, cy: int,
                   r_p: int, r_i: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = xx - cx, cy - yy
    r = np.hypot(dx, dy)
    annulus = (r > r_p) & (r <= r_i)

    ang = np.degrees(np.arctan2(dy, dx))            # -180..180, 0 = вправо
    side = np.minimum(np.abs(ang), 180 - np.abs(ang))  # угол до горизонтали
    r_mid = r_p + 0.6 * (r_i - r_p)
    ref = annulus & (side <= REF_ANGLE_DEG) & (r >= r_p + 0.1 * (r_i - r_p)) & (r <= r_mid)

    ref_vals = img[ref]
    if ref_vals.size < 50:
        ref_vals = img[annulus]
    if ref_vals.size == 0:
        return annulus, np.zeros_like(annulus)

    med = float(np.median(ref_vals))
    sigma = max(1.4826 * float(np.median(np.abs(ref_vals - med))), OCC_MIN_SIGMA)

    bright = ((img > med + OCC_K * sigma) | (img >= SPECULAR_LEVEL)).astype(np.uint8)
    dark = (img < med - OCC_K * sigma).astype(np.uint8)

    k3 = np.ones((3, 3), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k3)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dark = cv2.dilate(dark, k3, iterations=1)

    occluded = annulus & ((bright | dark) > 0)
    return annulus, occluded


def quality_score(img: np.ndarray, cx: int, cy: int,
                  r_p: int, r_i: int) -> tuple[float, float]:
    h, w = img.shape
    annulus, occluded = occlusion_mask(img, cx, cy, r_p, r_i)
    occ = float(occluded.sum() / max(annulus.sum(), 1))

    x1, y1 = max(cx - r_i, 0), max(cy - r_i, 0)
    x2, y2 = min(cx + r_i, w), min(cy + r_i, h)
    roi   = img[y1:y2, x1:x2]
    gx    = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy    = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    focus = min((gx ** 2 + gy ** 2).mean() / 5000.0, 1.0)

    score = float(np.clip(0.6 * (1 - occ) + 0.4 * focus, 0.0, 1.0))
    return score, occ


def process_one(img_path: Path, out_dir: Path,
                subject_key: str, sample_id: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id}.png"

    meta_base = {"subject_id": subject_key, "sample_id": sample_id,
                 "image_path": str(out_path), "source_path": str(img_path)}
    failed = {"quality_score": 0.0, "occlusion_ratio": 1.0, "low_quality": True,
              "pupil_found": False, "iris_found": False,
              "pupil_x": 0, "pupil_y": 0, "pupil_radius": 0, "iris_radius": 0}

    img = imread_unicode(img_path)
    if img is None:
        return {**meta_base, **failed}

    pupil = find_pupil(img)
    if pupil is None:
        fallback = cv2.resize(img, (RUBBER_W, RUBBER_H))
        imwrite_unicode(out_path, fallback)
        return {**meta_base, **failed}

    cx, cy, r_p = pupil
    r_i, iris_found = find_iris(img, cx, cy, r_p)

    qs, occ = quality_score(img, cx, cy, r_p, r_i)
    strip   = rubber_sheet(img, cx, cy, r_p, r_i)
    imwrite_unicode(out_path, strip)

    return {**meta_base,
            "quality_score":   round(qs, 4),
            "occlusion_ratio": round(occ, 4),
            "low_quality":     qs < QUALITY_THRESH,
            "pupil_found":     True,
            "iris_found":      iris_found,
            "pupil_x":         int(cx),
            "pupil_y":         int(cy),
            "pupil_radius":    int(r_p),
            "iris_radius":     int(r_i)}


def process_dataset(root: Path, ds_name: str) -> list[dict]:
    print(f"\n{'=' * 55}")
    print(f"[Iris] Датасет: {ds_name}")
    print(f"{'=' * 55}")

    if not root.exists():
        print(f"  Папка не найдена: {root}")
        return []

    all_meta = []
    subject_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    total = sum(
        len(list((sd / side).glob("*.jpg")) + list((sd / side).glob("*.bmp")))
        for sd in subject_dirs
        for side in ["L", "R"]
        if (sd / side).exists()
    )
    if total == 0:
        print(f"  В {root} не найдено папок вида <субъект>/L|R/*.jpg — "
              f"проверьте, нет ли лишнего уровня вложенности")
    done = 0

    for subj_dir in subject_dirs:
        subj_folder = subj_dir.name
        person_id = f"{ds_name}_{subj_folder}"
        for side in ["L", "R"]:
            side_dir = subj_dir / side
            if not side_dir.exists():
                continue
            subject_key = f"{person_id}_{side}"
            out_dir = OUTPUT_DIR / ds_name / subject_key
            imgs = sorted(set(side_dir.glob("*.jpg")) | set(side_dir.glob("*.bmp")))
            for img_path in imgs:
                rec = process_one(img_path, out_dir, subject_key, img_path.stem)
                rec["dataset"] = ds_name
                rec["person_id"] = person_id
                rec["eye"] = side
                all_meta.append(rec)
                done += 1
                if done % 200 == 0:
                    print(f"  {done}/{total} обработано...")

    lq = sum(1 for r in all_meta if r["low_quality"])
    no_pupil = sum(1 for r in all_meta if not r["pupil_found"])
    no_iris = sum(1 for r in all_meta if r["pupil_found"] and not r["iris_found"])
    n = max(len(all_meta), 1)
    print(f"\n  Итого: {len(all_meta)} образцов")
    print(f"  Зрачок не найден: {no_pupil} ({100 * no_pupil / n:.1f}%)")
    print(f"  Граница радужки не найдена (оценка 2.5·r_p): {no_iris} ({100 * no_iris / n:.1f}%)")
    print(f"  Низкое качество: {lq} ({100 * lq / n:.1f}%)")
    print(f"  Людей: {len({r['person_id'] for r in all_meta})}, "
          f"классов (глаз): {len({r['subject_id'] for r in all_meta})}")
    return all_meta


def save_metadata(meta: list[dict]) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nМетаданные сохранены → {META_PATH}")


if __name__ == "__main__":
    all_meta = []
    all_meta += process_dataset(INTERVAL_ROOT, "Interval")
    all_meta += process_dataset(THOUSAND_ROOT, "Thousand")
    if all_meta:
        save_metadata(all_meta)
    else:
        print("Изображения не найдены. Проверьте пути в начале файла.")
