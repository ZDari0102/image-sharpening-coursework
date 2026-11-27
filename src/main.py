import csv
import os
import sys
import time
import tracemalloc
from collections.abc import Callable

import cv2
import matplotlib

# Файловый backend позволяет сохранять рисунки без окон и графической сессии.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from convolution import clip_to_uint8, convolve2d_naive, convolve2d_opencv
from convolution_vectorized import convolve2d_vectorized
from kernels import KERNELS
from metrics import compute_metrics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MIN_PROFILE_NS = 200_000_000
CSV_FIELDS = (
    "image",
    "width",
    "height",
    "kernel",
    "variance_laplacian",
    "psnr_db",
    "ssim",
)

IMPLEMENTATIONS = (
    ("Naive", convolve2d_naive),
    ("NumPy", convolve2d_vectorized),
    ("OpenCV", convolve2d_opencv),
)


def load_grayscale(path: str) -> np.ndarray:
    if not os.path.exists(path):
        sys.exit(f"Файл не найден: {path}")
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        sys.exit(f"Не удалось прочитать: {path}")
    return image


def profile_execution(
    image: np.ndarray,
    kernel: np.ndarray,
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    label: str,
) -> tuple[np.ndarray, str]:
    """Измерить фильтрацию и вернуть отклик и подпись для рисунка."""
    # perf_counter учитывает ожидания. process_time суммирует время CPU
    # процесса в пользовательском и системном режимах, включая его потоки.
    start_wall = time.perf_counter_ns()
    start_cpu = time.process_time_ns()
    runs = 1
    result = func(image, kernel)
    wall_ns = time.perf_counter_ns() - start_wall
    # Несколько коротких вызовов уменьшают влияние дискретности таймера CPU.
    while wall_ns < MIN_PROFILE_NS:
        del result
        result = func(image, kernel)
        runs += 1
        wall_ns = time.perf_counter_ns() - start_wall
    cpu_ns = time.process_time_ns() - start_cpu

    # 100% соответствует полной занятости одного логического CPU.
    cpu_usage = f"{100 * cpu_ns / wall_ns:.1f}%"
    wall_ms = wall_ns / runs / 1_000_000
    cpu_ms = cpu_ns / runs / 1_000_000

    # Повторный вызов измеряет только пик отслеживаемых выделений памяти.
    # Трассировка выделений не входит в замеры времени и загрузки CPU.
    tracemalloc.start()
    try:
        func(image, kernel)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print(
        f"[{label}] Время: {wall_ms:.3f} мс | CPU-время: {cpu_ms:.3f} мс | "
        f"CPU: {cpu_usage} | Прогонов: {runs} | "
        f"Пик tracemalloc: {peak_bytes / 1024**2:.2f} МиБ",
        flush=True,
    )
    caption = f"{label}\nВремя: {wall_ms:.3f} мс\nCPU: {cpu_ms:.3f} мс ({cpu_usage})"
    return result, caption


def process_and_save(
    image: np.ndarray, kernel_name: str, kernel: np.ndarray, image_name: str
) -> np.ndarray:
    """Сохранить сравнение и вернуть изображение NumPy для оценки качества."""
    profiled = [
        profile_execution(image, kernel, func, label) for label, func in IMPLEMENTATIONS
    ]

    # Сравниваем отклики float32 до клиппинга, включая пиксели на границе.
    naive_result = profiled[0][0]
    for (label, _), (result, _) in zip(IMPLEMENTATIONS[1:], profiled[1:]):
        np.testing.assert_array_equal(
            result, naive_result, err_msg=f"{kernel_name}: {label}"
        )
    print("Проверка: все три отклика совпали попиксельно.", flush=True)

    # Преобразование для вывода, построение и запись рисунка идут вне замера.
    panels = [("Original", image)]
    panels.extend((caption, clip_to_uint8(result)) for result, caption in profiled)

    figure, axes = plt.subplots(1, len(panels), figsize=(16, 5), layout="constrained")
    for axis, (caption, pixels) in zip(axes, panels):
        axis.imshow(pixels, cmap="gray", vmin=0, vmax=255)
        axis.set_title(caption, fontsize=10)
        axis.axis("off")

    figure.suptitle(f"{image_name} | Kernel: {kernel_name}", fontsize=14)
    # Полное имя входного файла различает, например, photo.png и photo.jpg.
    output_path = os.path.join(OUTPUT_DIR, f"{image_name}_{kernel_name}_comparison.png")
    figure.savefig(output_path, dpi=150)
    # Освобождаем ресурсы рисунка перед следующей обработкой в пакете.
    plt.close(figure)
    print(f"Сохранено: {output_path}", flush=True)
    # Отклики трёх реализаций совпали. Для метрик используем результат NumPy
    # в uint8. После выхода из функции буферы float32 становятся ненужными.
    return panels[2][1]


def write_metrics_row(
    writer: csv.DictWriter,
    image_name: str,
    image: np.ndarray,
    kernel_name: str,
    values: dict[str, float],
) -> None:
    """Вывести метрики и записать строку CSV для импорта в таблицу."""
    print(
        f"[Метрики {kernel_name}] VoL: {values['variance_laplacian']:.4f} | "
        f"PSNR: {values['psnr_db']:.4f} дБ | SSIM: {values['ssim']:.6f}",
        flush=True,
    )
    # Разделитель «;», десятичная запятая и UTF-8 BOM удобны для русской версии Excel.
    # В CSV сохраняем точность float, округление применяется только в терминале.
    writer.writerow(
        {
            "image": image_name,
            "width": image.shape[1],
            "height": image.shape[0],
            "kernel": kernel_name,
            **{name: str(value).replace(".", ",") for name, value in values.items()},
        }
    )


def main() -> None:
    if not os.path.isdir(INPUT_DIR):
        sys.exit(f"Папка с изображениями не найдена: {INPUT_DIR}")

    image_names = [
        name
        for name in sorted(os.listdir(INPUT_DIR))
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        and os.path.isfile(os.path.join(INPUT_DIR, name))
    ]
    if not image_names:
        sys.exit(f"В папке нет поддерживаемых изображений: {INPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_path = os.path.join(OUTPUT_DIR, "quality_metrics.csv")
    # Каждый запуск формирует новую таблицу. Сбрасываем буфер после каждой строки,
    # чтобы результаты завершённых вычислений сохранялись и при долгом запуске.
    with open(metrics_path, "w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, delimiter=";")
        writer.writeheader()
        for index, image_name in enumerate(image_names, start=1):
            # Загружаем файлы по очереди, ограничивая объём входных данных в памяти.
            image = load_grayscale(os.path.join(INPUT_DIR, image_name))
            print(
                f"\nИзображение {index}/{len(image_names)}: {image_name} | {image.shape}",
                flush=True,
            )
            write_metrics_row(
                writer, image_name, image, "original", compute_metrics(image, image)
            )
            csv_file.flush()

            for name, kernel in KERNELS.items():
                print(f"\nОбработка ядром: {name}", flush=True)
                filtered = process_and_save(image, name, kernel, image_name)
                # Рабочие буферы метрик выделяются после завершения профилирования.
                values = compute_metrics(image, filtered)
                write_metrics_row(writer, image_name, image, name, values)
                csv_file.flush()

    print(f"\nРезультаты сохранены в: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Метрики качества сохранены в: {metrics_path}")


if __name__ == "__main__":
    main()
