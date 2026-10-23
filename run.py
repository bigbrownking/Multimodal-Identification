"""
Конвейер подготовки данных. Шаги можно передавать по одному или несколько подряд:

  python run.py --step all                     # всё по порядку
  python run.py --step sd302 splits corpus     # несколько шагов за раз

  Предобработка
    fingerprint      отпечатки: SOCOFing + NIST SD 302
    socofing         только SOCOFing
    sd302            только NIST SD 302 (нужен SD302_ROOT в .env)
    iris             радужка: CASIA-Iris-Interval + CASIA-Iris-Thousand
    iris_interval    только CASIA-Iris-Interval (остальные записи в метаданных сохраняются)
    iris_thousand    только CASIA-Iris-Thousand

  Разбиения и корпуса
    splits           train/val/test по людям для всех обработанных наборов
    corpus           оба мультимодальных корпуса + проверки:
                       multimodal_socofing  (SOCOFing × радужка — сценарий «отпечаток повреждён»)
                       multimodal_sd302     (SD 302 × радужка — основной)
    corpus_socofing  только multimodal_socofing
    corpus_sd302     только multimodal_sd302

  Отчёт
    analysis         статистика + графики качества по всем наборам (SOCOFing, SD 302, радужка)
    visualize        примеры до/после, проверка сегментации радужки

После предобработки любого набора перезапустите splits и corpus: разбиения детерминированы
(seed из .env), поэтому у других наборов они получатся прежними.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from fingerprint.preprocess_fingerprint import run_socofing, run_sd302
from iris.preprocess_iris import run_iris
from utils.build_splits import run_all as build_all_splits
from utils.build_corpus import run as build_corpus
from utils.visualize_samples import run as visualize_samples
from utils.analyse_data import run as analyse_data


def step_socofing():
    print("\n▶ Предобработка отпечатков: SOCOFing")
    run_socofing()


def step_sd302():
    print("\n▶ Предобработка отпечатков: NIST SD 302 (настоящие повторные прикладывания)")
    run_sd302()


def step_fingerprint():
    step_socofing()
    step_sd302()


def step_iris():
    print("\n▶ Предобработка радужки: Interval + Thousand")
    run_iris(("Interval", "Thousand"))


def step_iris_interval():
    print("\n▶ Предобработка радужки: CASIA-Iris-Interval")
    run_iris(("Interval",))


def step_iris_thousand():
    print("\n▶ Предобработка радужки: CASIA-Iris-Thousand")
    run_iris(("Thousand",))


def step_splits():
    print("\n▶ Разбиение train/val/test (по людям)")
    build_all_splits()


def step_corpus():
    print("\n▶ Мультимодальные корпуса виртуальных субъектов")
    build_corpus()


def step_corpus_socofing():
    print("\n▶ Мультимодальный корпус: SOCOFing × радужка")
    build_corpus(["multimodal_socofing"])


def step_corpus_sd302():
    print("\n▶ Мультимодальный корпус: SD 302 × радужка")
    build_corpus(["multimodal_sd302"])


def step_analysis():
    print("\n▶ Анализ данных и построение графиков (SOCOFing, SD 302, радужка)")
    analyse_data()


def step_visualize():
    print("\n▶ Визуальная проверка предобработки")
    visualize_samples()


# порядок здесь = порядок выполнения для --step all
PIPELINE = {
    "fingerprint": step_fingerprint,
    "iris":        step_iris,
    "splits":      step_splits,
    "corpus":      step_corpus,
    "analysis":    step_analysis,
    "visualize":   step_visualize,
}
SINGLE = {
    "socofing":        step_socofing,
    "sd302":           step_sd302,
    "iris_interval":   step_iris_interval,
    "iris_thousand":   step_iris_thousand,
    "corpus_socofing": step_corpus_socofing,
    "corpus_sd302":    step_corpus_sd302,
}
STEPS = {**PIPELINE, **SINGLE}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", nargs="+", default=["all"], choices=list(STEPS) + ["all"],
                        help="один или несколько шагов по порядку")
    args = parser.parse_args()

    steps = list(PIPELINE) if args.step == ["all"] else args.step
    for name in steps:
        STEPS[name]()

    print("\n✓ Готово")
