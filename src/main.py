import os
import sys
import time
import tracemalloc
from collections.abc import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np

from convolution import clip_to_uint8, convolve2d_naive, convolve2d_opencv
from convolution_vectorized import convolve2d_vectorized
from kernels import KERNELS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_DIR, "data", "1_high.png")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")
MIN_PROFILE_NS = 200_000_000

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


def process_and_save(image: np.ndarray, kernel_name: str, kernel: np.ndarray) -> None:
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

    figure.suptitle(f"Kernel: {kernel_name}", fontsize=14)
    figure.savefig(os.path.join(OUTPUT_DIR, f"{kernel_name}_comparison.png"), dpi=150)
    plt.show()
    plt.close(figure)


def main() -> None:
    image = load_grayscale(INPUT_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Загружено изображение: {image.shape}", flush=True)

    for name, kernel in KERNELS.items():
        print(f"\nОбработка ядром: {name}", flush=True)
        process_and_save(image, name, kernel)

    print(f"\nРезультаты сохранены в: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
