"""
  python run.py --step all           # полный пайплайн
  python run.py --step fingerprint   # только отпечатки
  python run.py --step iris          # только радужка
  python run.py --step splits        # train/val/test разбивка по людям
  python run.py --step corpus        # мультимодальный корпус виртуальных субъектов + проверки
  python run.py --step analysis      # статистика + графики
  python run.py --step visualize     # примеры до/после, проверка сегментации радужки
"""

import argparse
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from fingerprint.preprocess_fingerprint import process_socofing, save_metadata as fp_save
from iris.preprocess_iris import process_dataset, save_metadata as iris_save, INTERVAL_ROOT, THOUSAND_ROOT
from utils.build_splits import run_all as build_all_splits, METADATA_FINGERPRINT, METADATA_IRIS
from utils.build_corpus import run as build_corpus
from utils.visualize_samples import run as visualize_samples
from utils.analyse_data import (
    load as load_meta_a, print_summary,
    plot_quality, plot_lq_ratio, plot_occlusion, plot_samples_per_class,
)


def step_fingerprint():
    print("\n▶ Предобработка отпечатков пальцев")
    meta = process_socofing()
    if meta:
        fp_save(meta)


def step_iris():
    print("\n▶ Предобработка радужной оболочки")
    meta = []
    meta += process_dataset(INTERVAL_ROOT, "Interval")
    meta += process_dataset(THOUSAND_ROOT, "Thousand")
    if meta:
        iris_save(meta)


def step_splits():
    print("\n▶ Построение разбивки train/val/test (по людям)")
    build_all_splits()


def step_corpus():
    print("\n▶ Мультимодальный корпус виртуальных субъектов")
    build_corpus()


def step_analysis():
    print("\n▶ Анализ данных и построение графиков")
    fp   = load_meta_a(METADATA_FINGERPRINT)
    iris = load_meta_a(METADATA_IRIS)
    print_summary(fp, iris)
    plot_quality(fp, iris)
    plot_lq_ratio(fp, iris)
    plot_occlusion(iris)
    plot_samples_per_class(fp, iris)


def step_visualize():
    print("\n▶ Визуальная проверка предобработки")
    visualize_samples()


STEPS = {
    "fingerprint": step_fingerprint,
    "iris":        step_iris,
    "splits":      step_splits,
    "corpus":      step_corpus,
    "analysis":    step_analysis,
    "visualize":   step_visualize,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all",
                        choices=list(STEPS.keys()) + ["all"])
    args = parser.parse_args()

    if args.step == "all":
        for fn in STEPS.values():
            fn()
    else:
        STEPS[args.step]()

    print("\n✓ Готово")