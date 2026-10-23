import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SEED = int(os.getenv("SEED", "42"))
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES_PER_CLASS", "2"))
SPLITS = {
    "train": float(os.getenv("SPLIT_TRAIN")),
    "val": float(os.getenv("SPLIT_VAL")),
    "test": float(os.getenv("SPLIT_TEST")),
}

OUTPUT_SPLITS        = Path(os.getenv("OUTPUT_SPLITS"))
METADATA_FINGERPRINT = Path(os.getenv("METADATA_SOCOFING") or os.getenv("METADATA_FINGERPRINT")
                            or "data/processed/fingerprint_socofing_metadata.json")
METADATA_IRIS        = Path(os.getenv("METADATA_IRIS"))

SPLIT_NAMES = ("train", "val", "test")
REQUIRED_KEYS = ("person_id", "subject_id", "sample_id", "image_path", "quality_score", "low_quality")


def load_meta(path: Path) -> list[dict]:
    if not path.exists():
        print(f"Файл не найден: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    missing = [k for k in REQUIRED_KEYS if meta and k not in meta[0]]
    if missing:
        raise SystemExit(
            f"В {path} нет полей {missing}. Метаданные созданы старой версией "
            f"предобработки — перезапустите шаг fingerprint / iris."
        )
    return meta


def assign_persons(persons: list[str], seed: int = SEED) -> dict[str, str]:
    persons = sorted(persons)
    rng = random.Random(seed)
    rng.shuffle(persons)
    n = len(persons)
    n_train = round(n * SPLITS["train"])
    n_val = round(n * SPLITS["val"])
    out = {}
    for i, p in enumerate(persons):
        out[p] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_splits(records: list[dict], out_dir: Path, modality: str) -> dict:
    person_split = assign_persons(list({r["person_id"] for r in records}))

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_class[r["subject_id"]].append(r)

    splits = {s: [] for s in SPLIT_NAMES}
    hard, filtered = [], []
    for cls, recs in sorted(by_class.items()):
        split = person_split[recs[0]["person_id"]]
        good = [r for r in recs if not r["low_quality"]]
        bad = [r for r in recs if r["low_quality"]]
        for r in bad:
            filtered.append({**r, "modality": modality, "split": split, "reason": "low_quality"})
            if split == "test":
                hard.append({**r, "modality": modality, "split": "test_hard"})
        if len(good) < MIN_SAMPLES:
            for r in good:
                filtered.append({**r, "modality": modality, "split": split,
                                 "reason": f"class_has_<{MIN_SAMPLES}_good_samples"})
            continue
        splits[split] += [{**r, "modality": modality, "split": split} for r in good]

    for s in SPLIT_NAMES:
        write_csv(out_dir / f"{s}.csv", splits[s])
    write_csv(out_dir / "test_hard.csv", hard)
    write_csv(out_dir / "filtered_samples.csv", filtered)
    write_csv(out_dir / "person_split.csv",
              [{"person_id": p, "split": s} for p, s in sorted(person_split.items())])

    n_persons = len(person_split)
    summary = {}
    for s in SPLIT_NAMES:
        persons = {r["person_id"] for r in splits[s]}
        classes = {r["subject_id"] for r in splits[s]}
        share = sum(1 for v in person_split.values() if v == s) / max(n_persons, 1)
        summary[s] = {"persons": len(persons), "classes": len(classes),
                      "samples": len(splits[s]), "person_share": round(share, 4)}
        print(f"  {s:5s}: {len(splits[s]):6d} образцов, {len(classes):5d} классов, "
              f"{len(persons):4d} людей ({100 * share:.1f}% людей)")
    summary["test_hard"] = {"samples": len(hard)}
    summary["filtered"] = {"samples": len(filtered)}
    print(f"  test_hard: {len(hard)} образцов низкого качества; "
          f"исключено всего: {len(filtered)} → {out_dir / 'filtered_samples.csv'}")
    return summary


METADATA_SD302 = Path(os.getenv("METADATA_SD302", "data/processed/fingerprint_sd302_metadata.json"))

def run_sd302() -> None:
    print("\n[FINGERPRINT_SD302]")
    meta = load_meta(METADATA_SD302)
    if meta:
        make_splits(meta, OUTPUT_SPLITS / "fingerprint_sd302", "fingerprint_sd302")
    else:
        print("  Нет данных — сначала запустите предобработку SD302")


def run_all() -> None:
    for modality, meta_path in [("fingerprint_socofing", METADATA_FINGERPRINT), ("iris", METADATA_IRIS)]:
        print(f"\n[{modality.upper()}]")
        meta = load_meta(meta_path)
        if meta:
            make_splits(meta, OUTPUT_SPLITS / modality, modality)
        else:
            print(f"  Нет данных — сначала запустите предобработку ({modality})")
    if METADATA_SD302.exists():
        run_sd302()


if __name__ == "__main__":
    print("=" * 55)
    print("Построение train/val/test разбивки")
    print("=" * 55)
    run_all()
    print("\nГотово!")
