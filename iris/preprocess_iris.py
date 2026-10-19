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


PUPIL_R_MIN, PUPIL_R_MAX = 12, 80
LIMBUS_MIN_CONTRAST = 6.0


def _pupil_dark_ok(img: np.ndarray, cx: int, cy: int, r: int) -> bool:
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    d = (xx - cx) ** 2 + (yy - cy) ** 2
    inside = img[d <= (0.7 * r) ** 2]
    ring = img[(d > (1.3 * r) ** 2) & (d <= (1.8 * r) ** 2)]
    if inside.size == 0 or ring.size == 0:
        return False
    return float(np.median(inside)) + 20 < float(np.median(ring))


def _blob_circle(c, fill: float) -> tuple[int, int, int]:
    if fill >= 0.85:
        m = cv2.moments(c)
        cx, cy, r = m["m10"] / m["m00"], m["m01"] / m["m00"], np.sqrt(m["m00"] / np.pi)
    else:
        (cx, cy), r = cv2.minEnclosingCircle(c)
    return int(round(cx)), int(round(cy)), int(round(r))


def _edge_step(img: np.ndarray, cx: int, cy: int, r: int) -> float:
    h, w = img.shape
    x1, x2 = max(cx - int(1.4 * r) - 1, 0), min(cx + int(1.4 * r) + 2, w)
    y1, y2 = max(cy - int(1.4 * r) - 1, 0), min(cy + int(1.4 * r) + 2, h)
    patch = img[y1:y2, x1:x2]
    yy, xx = np.ogrid[y1:y2, x1:x2]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1)
    inner = patch[(d >= 0.75) & (d <= 0.95)]
    outer = patch[(d >= 1.05) & (d <= 1.3)]
    if inner.size < 10 or outer.size < 10:
        return -1.0
    return float(np.median(outer)) - float(np.median(inner))


def find_pupil_dark(img: np.ndarray) -> tuple[int, int, int] | None:
    blur = cv2.medianBlur(img, 5)
    lo = float(np.percentile(blur, 0.5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    best = None
    for off in (10, 15, 20, 25, 30, 40, 50, 60, 75, 90):
        mask = (blur <= min(lo + off, 130)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area = cv2.contourArea(c)
            if not (np.pi * PUPIL_R_MIN ** 2 <= area <= np.pi * PUPIL_R_MAX ** 2):
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            fill = area / (np.pi * r * r)
            solidity = area / max(cv2.contourArea(cv2.convexHull(c)), 1.0)
            if fill < 0.55 or solidity < 0.85:
                continue
            cand = _blob_circle(c, fill)
            if not _pupil_dark_ok(img, *cand):
                continue
            score = _edge_step(img, *cand) * min(fill / 0.85, 1.0)
            if best is None or score > best[0]:
                best = (score, cand)
    return best[1] if best else None


def find_pupil_hough(img: np.ndarray) -> tuple[int, int, int] | None:
    blurred = cv2.GaussianBlur(img, (7, 7), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=30,
                               param1=100, param2=15,
                               minRadius=PUPIL_R_MIN, maxRadius=PUPIL_R_MAX)
    if circles is None:
        return None
    for c in np.round(circles[0]).astype(int)[:10]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        if _pupil_dark_ok(img, cx, cy, r):
            return cx, cy, r
    return None


def find_pupil(img: np.ndarray) -> tuple[tuple[int, int, int], str] | None:
    p = find_pupil_dark(img)
    if p is not None:
        return p, "dark_blob"
    p = find_pupil_hough(img)
    if p is not None:
        return p, "hough"
    return None


def _limbus_peaks(blur: np.ndarray, cx: int, cy: int, radii: np.ndarray,
                  angles_deg: np.ndarray, k: int = 6) -> list[tuple[int, float]]:
    h, w = blur.shape
    a = np.radians(angles_deg)
    X = cx + radii[:, None] * np.cos(a)[None, :]
    Y = cy - radii[:, None] * np.sin(a)[None, :]
    valid = (X >= 0) & (X <= w - 1) & (Y >= 0) & (Y <= h - 1)
    vals = cv2.remap(blur, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR)
    vals = np.where(valid, vals, np.nan)
    ok = valid.mean(axis=1) >= 0.5
    n = int(np.argmin(ok)) if not ok.all() else len(ok)
    if n < 12:
        return []
    prof = np.nanmedian(vals[:n], axis=1)
    g = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2); g /= g.sum()
    prof = np.convolve(np.pad(prof, 6, mode="edge"), g, mode="valid")
    d = prof[4:] - prof[:-4]
    peaks = [i for i in range(1, len(d) - 1)
             if d[i] >= d[i - 1] and d[i] >= d[i + 1] and d[i] >= LIMBUS_MIN_CONTRAST]
    peaks.sort(key=lambda i: -d[i])
    return [(int(radii[i + 2]), float(d[i])) for i in peaks[:k]]


def limbus_candidates(img: np.ndarray, cx: int, cy: int, r_p: int) -> dict:
    h, w = img.shape
    blur = cv2.GaussianBlur(img, (5, 5), 1.5).astype(np.float32)
    r_min = int(r_p * 1.4) + 4
    r_max = int(min(r_p * 6.0, 0.5 * min(h, w)))
    if r_max <= r_min + 12:
        return {"right": [], "left": []}
    radii = np.arange(r_min, r_max, dtype=np.float32)
    return {"right": _limbus_peaks(blur, cx, cy, radii, np.arange(-30, 11, 2)),
            "left": _limbus_peaks(blur, cx, cy, radii, np.arange(170, 211, 2))}


def choose_iris(cx: int, cy: int, r_p: int, cands: dict,
                r_range: tuple[float, float] | None = None,
                fallback_r: float | None = None) -> tuple[int, int, int, bool, bool]:
    def best(side):
        c = [p for p in cands.get(side, [])
             if r_range is None or r_range[0] <= p[0] <= r_range[1]]
        return max(c, key=lambda p: p[1]) if c else None

    right, left = best("right"), best("left")
    if right and left and abs(right[0] - left[0]) <= 0.3 * (right[0] + left[0]) / 2:
        r_i = (right[0] + left[0]) / 2
        ix = cx + (right[0] - left[0]) / 2
        return int(round(ix)), cy, int(round(r_i)), True, True
    if right or left:
        b = max([s for s in (right, left) if s], key=lambda p: p[1])
        return cx, cy, int(b[0]), True, False
    r = fallback_r if fallback_r else 2.5 * r_p
    return cx, cy, int(round(r)), False, False


def find_iris(img: np.ndarray, cx: int, cy: int, r_p: int,
              r_range: tuple[float, float] | None = None) -> tuple[int, int, int, bool]:
    ix, iy, r_i, found, _ = choose_iris(cx, cy, r_p, limbus_candidates(img, cx, cy, r_p), r_range)
    return ix, iy, r_i, found


def rubber_sheet(img: np.ndarray, cx: int, cy: int, r_p: int,
                 ix: int, iy: int, r_i: int) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, RUBBER_W, endpoint=False)
    rho = np.linspace(0, 1, RUBBER_H)[:, None]
    px, py = cx + r_p * np.cos(theta), cy - r_p * np.sin(theta)
    qx, qy = ix + r_i * np.cos(theta), iy - r_i * np.sin(theta)
    X = ((1 - rho) * px[None, :] + rho * qx[None, :]).astype(np.float32)
    Y = ((1 - rho) * py[None, :] + rho * qy[None, :]).astype(np.float32)
    strip = cv2.remap(img, X, Y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 16))
    return clahe.apply(strip)


def occlusion_mask(img: np.ndarray, cx: int, cy: int, r_p: int, r_i: int,
                   ix: int | None = None, iy: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    ix = cx if ix is None else ix
    iy = cy if iy is None else iy
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = xx - cx, cy - yy
    r = np.hypot(dx, dy)
    annulus = (r > r_p) & (np.hypot(xx - ix, yy - iy) <= r_i)

    ang = np.degrees(np.arctan2(dy, dx))
    right = (ang >= -REF_ANGLE_DEG) & (ang <= 10)
    left = (ang >= 170) | (ang <= -180 + REF_ANGLE_DEG)
    r_mid = r_p + 0.6 * (r_i - r_p)
    ref = annulus & (right | left) & (r >= r_p + 0.1 * (r_i - r_p)) & (r <= r_mid)

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


def quality_score(img: np.ndarray, cx: int, cy: int, r_p: int, r_i: int,
                  ix: int | None = None, iy: int | None = None) -> tuple[float, float]:
    ix = cx if ix is None else ix
    iy = cy if iy is None else iy
    h, w = img.shape
    annulus, occluded = occlusion_mask(img, cx, cy, r_p, r_i, ix, iy)
    occ = float(occluded.sum() / max(annulus.sum(), 1))

    x1, y1 = max(ix - r_i, 0), max(iy - r_i, 0)
    x2, y2 = min(ix + r_i, w), min(iy + r_i, h)
    roi   = img[y1:y2, x1:x2]
    gx    = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy    = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    focus = min((gx ** 2 + gy ** 2).mean() / 5000.0, 1.0)

    score = float(np.clip(0.6 * (1 - occ) + 0.4 * focus, 0.0, 1.0))
    return score, occ


IRIS_PRIOR_RANGE = (0.78, 1.25)


def _failed(meta_base: dict) -> dict:
    return {**meta_base, "quality_score": 0.0, "occlusion_ratio": 1.0, "low_quality": True,
            "pupil_found": False, "pupil_method": "none", "iris_found": False,
            "pupil_x": 0, "pupil_y": 0, "pupil_radius": 0,
            "iris_x": 0, "iris_y": 0, "iris_radius": 0}


def process_dataset(root: Path, ds_name: str) -> list[dict]:
    print(f"\n{'=' * 55}")
    print(f"[Iris] Датасет: {ds_name}")
    print(f"{'=' * 55}")

    if not root.exists():
        print(f"  Папка не найдена: {root}")
        return []

    items = []
    for subj_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        person_id = f"{ds_name}_{subj_dir.name}"
        for side in ["L", "R"]:
            side_dir = subj_dir / side
            if not side_dir.exists():
                continue
            for img_path in sorted(set(side_dir.glob("*.jpg")) | set(side_dir.glob("*.bmp"))):
                items.append((img_path, person_id, side))
    total = len(items)
    if total == 0:
        print(f"  В {root} не найдено папок вида <субъект>/L|R/*.jpg — "
              f"проверьте, нет ли лишнего уровня вложенности")
        return []

    # ---- проход 1: зрачок и кандидаты лимба
    stage1 = []
    for n, (img_path, person_id, side) in enumerate(items, 1):
        img = imread_unicode(img_path)
        found = find_pupil(img) if img is not None else None
        if found is None:
            stage1.append(None)
        else:
            (cx, cy, r_p), method = found
            cands = limbus_candidates(img, cx, cy, r_p)
            stage1.append({"pupil": (cx, cy, r_p), "method": method, "cands": cands,
                           "free": choose_iris(cx, cy, r_p, cands)})
        if n % 500 == 0:
            print(f"  проход 1: {n}/{total}")

    agreed = [s["free"][2] for s in stage1 if s and s["free"][4]]
    r0 = float(np.median(agreed)) if len(agreed) >= 30 else None
    if r0:
        lo, hi = IRIS_PRIOR_RANGE[0] * r0, IRIS_PRIOR_RANGE[1] * r0
        print(f"  Типичный радиус радужки R0 = {r0:.0f} px (по {len(agreed)} снимкам), "
              f"коридор {lo:.0f}–{hi:.0f} px")
    else:
        print("  Мало согласованных снимков — радиус радужки без ограничения по датасету")

    # ---- проход 2: окончательная граница, качество, нормализация
    all_meta = []
    for n, ((img_path, person_id, side), s1) in enumerate(zip(items, stage1), 1):
        subject_key = f"{person_id}_{side}"
        out_dir = OUTPUT_DIR / ds_name / subject_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{img_path.stem}.png"
        meta_base = {"subject_id": subject_key, "sample_id": img_path.stem,
                     "image_path": str(out_path), "source_path": str(img_path),
                     "dataset": ds_name, "person_id": person_id, "eye": side}
        img = imread_unicode(img_path)

        cx = cy = r_p = None
        if s1 is not None and img is not None:
            cx, cy, r_p = s1["pupil"]
            if r0 and r_p >= 0.7 * r0:
                s1 = None
        if s1 is None or img is None:
            if img is not None:
                imwrite_unicode(out_path, cv2.resize(img, (RUBBER_W, RUBBER_H)))
            all_meta.append(_failed(meta_base))
            continue

        rng = (IRIS_PRIOR_RANGE[0] * r0, IRIS_PRIOR_RANGE[1] * r0) if r0 else None
        ix, iy, r_i, iris_found, _ = choose_iris(cx, cy, r_p, s1["cands"], rng, r0)
        qs, occ = quality_score(img, cx, cy, r_p, r_i, ix, iy)
        imwrite_unicode(out_path, rubber_sheet(img, cx, cy, r_p, ix, iy, r_i))
        all_meta.append({**meta_base,
                         "quality_score":   round(qs, 4),
                         "occlusion_ratio": round(occ, 4),
                         "low_quality":     qs < QUALITY_THRESH,
                         "pupil_found":     True,
                         "pupil_method":    s1["method"],
                         "iris_found":      iris_found,
                         "pupil_x":         int(cx),
                         "pupil_y":         int(cy),
                         "pupil_radius":    int(r_p),
                         "iris_x":          int(ix),
                         "iris_y":          int(iy),
                         "iris_radius":     int(r_i),
                         "iris_prior_radius": round(r0, 1) if r0 else None})
        if n % 500 == 0:
            print(f"  проход 2: {n}/{total}")

    lq = sum(1 for r in all_meta if r["low_quality"])
    no_pupil = sum(1 for r in all_meta if not r["pupil_found"])
    no_iris = sum(1 for r in all_meta if r["pupil_found"] and not r["iris_found"])
    hough = sum(1 for r in all_meta if r.get("pupil_method") == "hough")
    n = max(len(all_meta), 1)
    print(f"\n  Итого: {len(all_meta)} образцов")
    print(f"  Зрачок не найден: {no_pupil} ({100 * no_pupil / n:.1f}%)")
    print(f"  Граница радужки не найдена в коридоре (взят R0): {no_iris} ({100 * no_iris / n:.1f}%)")
    print(f"  Зрачок найден запасным методом (Хаф): {hough} ({100 * hough / n:.1f}%)")
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
