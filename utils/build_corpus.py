import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SEED = int(os.getenv("SEED", "42"))
OUTPUT_SPLITS = Path(os.getenv("OUTPUT_SPLITS"))
FP_DIR = OUTPUT_SPLITS / "fingerprint"
IRIS_DIR = OUTPUT_SPLITS / "iris"
MM_DIR = OUTPUT_SPLITS / "multimodal"

SPLIT_NAMES = ("train", "val", "test")


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def group_by_class(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["subject_id"]].append(r)
    return out


def pair_row(vid: str, split: str, fp: dict, ir: dict) -> dict:
    return {
        "virtual_id":      vid,
        "split":           split,
        "fp_class":        fp["subject_id"],
        "fp_person":       fp["person_id"],
        "fp_sample":       fp["sample_id"],
        "fp_subset":       fp.get("dataset", ""),
        "fp_alteration":   fp.get("alteration", ""),
        "fp_image_path":   fp["image_path"],
        "fp_image_path_clahe": fp.get("image_path_clahe", ""),
        "fp_quality":      fp["quality_score"],
        "iris_class":      ir["subject_id"],
        "iris_person":     ir["person_id"],
        "iris_sample":     ir["sample_id"],
        "iris_dataset":    ir.get("dataset", ""),
        "iris_eye":        ir.get("eye", ""),
        "iris_image_path": ir["image_path"],
        "iris_quality":    ir["quality_score"],
        "iris_occlusion":  ir.get("occlusion_ratio", ""),
    }


def build() -> dict:
    rng = random.Random(SEED)
    fp_split = {s: read_csv(FP_DIR / f"{s}.csv") for s in SPLIT_NAMES}
    ir_split = {s: read_csv(IRIS_DIR / f"{s}.csv") for s in SPLIT_NAMES}
    if not any(fp_split.values()) or not any(ir_split.values()):
        raise SystemExit("Нет мономодальных разбиений — сначала запустите шаг splits.")
    fp_hard = group_by_class(read_csv(FP_DIR / "test_hard.csv"))
    ir_hard = group_by_class(read_csv(IRIS_DIR / "test_hard.csv"))

    pairing, rows, hard_rows = [], {s: [] for s in SPLIT_NAMES}, []
    unpaired = {}
    counter = 0

    for s in SPLIT_NAMES:
        fp_cls = group_by_class(fp_split[s])
        ir_cls = group_by_class(ir_split[s])
        fp_keys, ir_keys = sorted(fp_cls), sorted(ir_cls)
        rng.shuffle(fp_keys)
        rng.shuffle(ir_keys)
        n = min(len(fp_keys), len(ir_keys))
        unpaired[s] = {"fingerprint_classes": len(fp_keys) - n, "iris_classes": len(ir_keys) - n}

        for fk, ik in zip(fp_keys[:n], ir_keys[:n]):
            counter += 1
            vid = f"V{counter:05d}"
            fps, irs = fp_cls[fk][:], ir_cls[ik][:]
            rng.shuffle(fps)
            rng.shuffle(irs)
            k = min(len(fps), len(irs))
            for fp, ir in zip(fps[:k], irs[:k]):
                rows[s].append(pair_row(vid, s, fp, ir))
            pairing.append({
                "virtual_id": vid, "split": s,
                "fp_class": fk, "fp_person": fps[0]["person_id"],
                "iris_class": ik, "iris_person": irs[0]["person_id"],
                "iris_dataset": irs[0].get("dataset", ""), "n_pairs": k,
            })

            if s == "test":
                for fp in fp_hard.get(fk, []):
                    hard_rows.append({**pair_row(vid, "test_hard", fp, rng.choice(irs)),
                                      "hard_modality": "fingerprint"})
                for ir in ir_hard.get(ik, []):
                    hard_rows.append({**pair_row(vid, "test_hard", rng.choice(fps), ir),
                                      "hard_modality": "iris"})

    for s in SPLIT_NAMES:
        write_csv(MM_DIR / f"{s}.csv", rows[s])
    write_csv(MM_DIR / "test_hard.csv", hard_rows)
    write_csv(MM_DIR / "pairing.csv", pairing)

    total_v = len(pairing)
    summary = {"seed": SEED, "splits": {}, "unpaired": unpaired}
    print(f"\n  Виртуальных субъектов: {total_v}")
    for s in SPLIT_NAMES:
        nv = sum(1 for p in pairing if p["split"] == s)
        by_ds = defaultdict(int)
        for p in pairing:
            if p["split"] == s:
                by_ds[p["iris_dataset"]] += 1
        summary["splits"][s] = {"virtual_subjects": nv, "pairs": len(rows[s]),
                                "share": round(nv / max(total_v, 1), 4),
                                "iris_dataset": dict(by_ds)}
        print(f"  {s:5s}: {nv:5d} субъектов ({100 * nv / max(total_v, 1):.1f}%), "
              f"{len(rows[s]):6d} пар; радужка: {dict(by_ds)}; "
              f"без пары осталось: {unpaired[s]['fingerprint_classes']} классов пальцев, "
              f"{unpaired[s]['iris_classes']} классов глаз")
    summary["splits"]["test_hard"] = {"pairs": len(hard_rows)}
    print(f"  test_hard: {len(hard_rows)} пар с образцом низкого качества")
    return summary


def verify() -> dict:
    checks = {}

    def sets(rows, key):
        return {s: {r[key] for r in rows[s]} for s in SPLIT_NAMES}

    def disjoint(name, by_split):
        for i, a in enumerate(SPLIT_NAMES):
            for b in SPLIT_NAMES[i + 1:]:
                inter = by_split[a] & by_split[b]
                assert not inter, f"{name}: пересечение {a}/{b}: {sorted(inter)[:5]}"
        checks[name] = "ok"

    # 1. Мономодальные разбиения: люди и классы не пересекаются
    split_of_image = {}
    for mod, d in [("fingerprint", FP_DIR), ("iris", IRIS_DIR)]:
        rows = {s: read_csv(d / f"{s}.csv") for s in SPLIT_NAMES}
        disjoint(f"{mod}: люди", sets(rows, "person_id"))
        disjoint(f"{mod}: классы", sets(rows, "subject_id"))
        for s in SPLIT_NAMES:
            for r in rows[s]:
                split_of_image[r["image_path"]] = s
        hard_persons = {r["person_id"] for r in read_csv(d / "test_hard.csv")}
        assert hard_persons <= {r["person_id"] for r in read_csv(d / "person_split.csv")
                                if r["split"] == "test"}, f"{mod}: test_hard не из test"
        checks[f"{mod}: test_hard только из test"] = "ok"

    # 2. Мультимодальный корпус
    mm = {s: read_csv(MM_DIR / f"{s}.csv") for s in SPLIT_NAMES}
    disjoint("multimodal: виртуальные субъекты", sets(mm, "virtual_id"))
    disjoint("multimodal: люди (отпечатки)", sets(mm, "fp_person"))
    disjoint("multimodal: люди (радужка)", sets(mm, "iris_person"))

    pairing = read_csv(MM_DIR / "pairing.csv")
    for col in ("fp_class", "iris_class", "virtual_id"):
        vals = [p[col] for p in pairing]
        assert len(vals) == len(set(vals)), f"pairing: {col} используется больше одного раза"
    checks["pairing: каждый класс в одном виртуальном субъекте"] = "ok"

    for s in SPLIT_NAMES:
        for r in mm[s]:
            assert split_of_image.get(r["fp_image_path"]) == s, f"{r['fp_image_path']} не из {s}"
            assert split_of_image.get(r["iris_image_path"]) == s, f"{r['iris_image_path']} не из {s}"
    checks["multimodal: образцы взяты из своей выборки"] = "ok"

    test_v = {r["virtual_id"] for r in mm["test"]}
    hard_v = {r["virtual_id"] for r in read_csv(MM_DIR / "test_hard.csv")}
    assert hard_v <= test_v, "test_hard содержит субъектов не из test"
    checks["multimodal: test_hard только из test"] = "ok"

    print("\n  Проверки:")
    for k, v in checks.items():
        print(f"    [{v}] {k}")
    return checks


def run() -> None:
    summary = build()
    summary["checks"] = verify()
    MM_DIR.mkdir(parents=True, exist_ok=True)
    with open(MM_DIR / "corpus_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Сводка → {MM_DIR / 'corpus_summary.json'}")


if __name__ == "__main__":
    print("=" * 55)
    print("Мультимодальный корпус виртуальных субъектов")
    print("=" * 55)
    run()
