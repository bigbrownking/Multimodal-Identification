import json
import csv
import os
import random
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

SEED = 42
SPLITS = {
    "train": float(os.getenv("SPLIT_TRAIN")),
    "val": float(os.getenv("SPLIT_VAL")),
    "test": float(os.getenv("SPLIT_TEST")),
}

OUTPUT_SPLITS       = Path(os.getenv("OUTPUT_SPLITS"))
METADATA_FINGERPRINT = Path(os.getenv("METADATA_FINGERPRINT"))
METADATA_IRIS        = Path(os.getenv("METADATA_IRIS"))


def load_meta(path: Path) -> list[dict]:
    if not path.exists():
        print(f"Файл не найден: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_splits(records: list[dict],
                out_dir: Path,
                modality: str) -> None:
    by_subj: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if not r.get("low_quality", False):
            by_subj[r["subject_id"]].append(r)

    subjects = sorted(by_subj.keys())
    random.seed(SEED)
    random.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])

    assignment = {}
    for s in subjects[:n_train]:
        assignment[s] = "train"
    for s in subjects[n_train:n_train + n_val]:
        assignment[s] = "val"
    for s in subjects[n_train + n_val:]:
        assignment[s] = "test"

    split_records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for subj, recs in by_subj.items():
        split_name = assignment.get(subj, "test")
        for r in recs:
            split_records[split_name].append({**r, "modality": modality})

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, recs in split_records.items():
        if not recs:
            continue
        out_path = out_dir / f"{split_name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            writer.writeheader()
            writer.writerows(recs)

        subj_count = len({str(r["subject_id"]) for r in recs})
        print(f"  {split_name:5s}: {len(recs):5d} образцов, "
              f"{subj_count:4d} субъектов → {out_path}")


if __name__ == "__main__":
    print("=" * 55)
    print("Построение train/val/test разбивки")
    print("=" * 55)

    for modality, meta_path, out_dir in [
        ("fingerprint", METADATA_FINGERPRINT, OUTPUT_SPLITS / "fingerprint"),
        ("iris",        METADATA_IRIS,        OUTPUT_SPLITS / "iris"),
    ]:
        print(f"\n[{modality.upper()}]")
        meta = load_meta(meta_path)
        if meta:
            make_splits(meta, out_dir, modality)
        else:
            print(f"  Нет данных — сначала запустите preprocess_{modality}.py")

    print("\nГотово!")