import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset

load_dotenv()
# Каждый процесс загрузки сам по себе однопоточный: иначе 4 процесса × потоки OpenCV
# дерутся за 6 ядер и замедляют друг друга
cv2.setNumThreads(0)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_SPLITS = ROOT / os.getenv("OUTPUT_SPLITS", "data/splits")
OUTPUT_ANALYSIS = ROOT / os.getenv("OUTPUT_ANALYSIS", "data/analysis_plots")

FP_SIZE = (224, 224)
IRIS_SIZE = (64, 512)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
FP_INPUT = os.getenv("FP_INPUT", "gabor")


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def read_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть {path}")
    if img.shape != size:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return img


def jitter(img: np.ndarray, rng: np.random.Generator, a: float, b: float) -> np.ndarray:
    alpha = rng.uniform(1 - a, 1 + a)
    beta = rng.uniform(-b, b)
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def border_value(img: np.ndarray, width: int = 4) -> int:
    """Цвет фона = медиана полосы по краю снимка (у SOCOFing после среза рамки — белый).
    Раньше бралась медиана всего снимка -> серые клинья в углах после поворота."""
    ring = np.concatenate([img[:width].ravel(), img[-width:].ravel(),
                           img[:, :width].ravel(), img[:, -width:].ravel()])
    return int(np.median(ring))


def augment_fp(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), rng.uniform(0.95, 1.05))
    m[:, 2] += rng.uniform(-0.05, 0.05, size=2) * (w, h)
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border_value(img))
    return jitter(img, rng, 0.10, 10)


def augment_iris(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = np.roll(img, int(rng.integers(-16, 17)), axis=1)
    return jitter(img, rng, 0.15, 15)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).float().div_(255.0).unsqueeze(0).repeat(3, 1, 1)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Нет манифеста {path} — запустите run.py --step splits/corpus")
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _rng() -> np.random.Generator:
    return np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))


class BiometricDataset(Dataset):
    def __init__(self, modality: str, split: str, train: bool | None = None,
                 label_map: dict[str, int] | None = None, fp_input: str = FP_INPUT):
        assert modality in ("fingerprint", "iris")
        assert fp_input in ("gabor", "clahe")
        self.modality = modality
        self.split = split
        self.train = (split == "train") if train is None else train
        self.size = FP_SIZE if modality == "fingerprint" else IRIS_SIZE
        rows = read_csv(OUTPUT_SPLITS / modality / f"{split}.csv")
        col = "image_path_clahe" if modality == "fingerprint" and fp_input == "clahe" else "image_path"
        self.paths = [resolve(r[col]) for r in rows]
        self.classes = [r["subject_id"] for r in rows]
        self.rows = rows
        self.label_map = label_map or {c: i for i, c in enumerate(sorted(set(self.classes)))}
        self.labels = [self.label_map[c] for c in self.classes]

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, i: int, augment: bool) -> np.ndarray:
        img = read_gray(self.paths[i], self.size)
        if augment:
            fn = augment_fp if self.modality == "fingerprint" else augment_iris
            img = fn(img, _rng())
        return img

    def __getitem__(self, i: int):
        return to_tensor(self.load(i, self.train)), self.labels[i]


class MultimodalDataset(Dataset):
    def __init__(self, split: str, train: bool | None = None,
                 label_map: dict[str, int] | None = None, fp_input: str = FP_INPUT):
        self.split = split
        self.train = (split == "train") if train is None else train
        self.rows = read_csv(OUTPUT_SPLITS / "multimodal" / f"{split}.csv")
        col = "fp_image_path_clahe" if fp_input == "clahe" else "fp_image_path"
        self.fp_paths = [resolve(r[col]) for r in self.rows]
        self.iris_paths = [resolve(r["iris_image_path"]) for r in self.rows]
        vids = [r["virtual_id"] for r in self.rows]
        self.label_map = label_map or {v: i for i, v in enumerate(sorted(set(vids)))}
        self.labels = [self.label_map[v] for v in vids]

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        fp = read_gray(self.fp_paths[i], FP_SIZE)
        ir = read_gray(self.iris_paths[i], IRIS_SIZE)
        if self.train:
            rng = _rng()
            fp, ir = augment_fp(fp, rng), augment_iris(ir, rng)
        return to_tensor(fp), to_tensor(ir), self.labels[i]


def make_loader(ds: Dataset, batch_size: int = 32, shuffle: bool | None = None,
                num_workers: int = 2) -> DataLoader:
    shuffle = getattr(ds, "train", False) if shuffle is None else shuffle
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle,
                      persistent_workers=num_workers > 0)


# ---------- проверка конвейера ----------
def _check_files(paths: list[Path]) -> int:
    return sum(1 for p in paths if not p.exists())


def _augmentation_grid() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for mod in ("fingerprint", "iris"):
        ds = BiometricDataset(mod, "train")
        idx = np.random.default_rng(0).choice(len(ds), 3, replace=False)
        for i in idx:
            rows.append([ds.load(int(i), False)] + [ds.load(int(i), True) for _ in range(4)])
    fig, axes = plt.subplots(len(rows), 5, figsize=(14, 2.0 * len(rows)),
                             gridspec_kw={"height_ratios": [3] * 3 + [1] * 3})
    for r, imgs in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("original" if c == 0 else f"augmented #{c}", fontsize=9)
    fig.suptitle("Training augmentations: fingerprint (top), iris strip (bottom)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    OUTPUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
    p = OUTPUT_ANALYSIS / "augmentation_examples.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def check(num_workers: int = 2, batch_size: int = 32) -> None:
    print(f"torch {torch.__version__}, CUDA доступна: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    print(f"Вход для отпечатков: FP_INPUT={FP_INPUT}")
    for mod in ("fingerprint", "iris"):
        print(f"\n[{mod.upper()}]")
        for split in ("train", "val", "test"):
            ds = BiometricDataset(mod, split)
            missing = _check_files(ds.paths)
            if mod == "fingerprint":
                missing += _check_files(BiometricDataset(mod, split, fp_input="clahe").paths)
            print(f"  {split:5s}: {len(ds):6d} образцов, {ds.num_classes:5d} классов, "
                  f"нет файлов: {missing}")

    print("\n[MULTIMODAL]")
    for split in ("train", "val", "test"):
        ds = MultimodalDataset(split)
        missing = _check_files(ds.fp_paths) + _check_files(ds.iris_paths)
        print(f"  {split:5s}: {len(ds):6d} пар, {ds.num_classes:5d} субъектов, нет файлов: {missing}")

    print(f"\n[ЗАГРУЗЧИК] batch_size={batch_size}, num_workers={num_workers}")
    for name, ds in [("fingerprint", BiometricDataset("fingerprint", "train")),
                     ("iris", BiometricDataset("iris", "train")),
                     ("multimodal", MultimodalDataset("train"))]:
        loader = make_loader(ds, batch_size=batch_size, num_workers=num_workers)
        t0 = time.perf_counter()
        it = iter(loader)
        batches = [next(it) for _ in range(5)]
        dt = (time.perf_counter() - t0) / 5
        b = batches[0]
        shapes = ", ".join(str(tuple(x.shape)) for x in b[:-1])
        x = b[0]
        print(f"  {name:11s}: {shapes}, labels {tuple(b[-1].shape)}, "
              f"значения [{x.min():.2f}, {x.max():.2f}], {1000 * dt:.0f} мс/батч")

    _augmentation_grid()
    print("\n✓ Конвейер загрузки работает")


if __name__ == "__main__":
    check()