import os
import sys
import time
import tracemalloc

import cv2
import matplotlib.pyplot as plt
import numpy as np

from convolution import clip_to_uint8, convolve2d_naive, convolve2d_opencv
from kernels import KERNELS

# Получаем директорию, где лежит main.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

INPUT_PATH = os.path.join(PROJECT_DIR, "data", "2_low.png")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")


def load_grayscale(path: str) -> np.ndarray:
    if not os.path.exists(path):
        sys.exit(f"Файл не найден: {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"Не удалось прочитать: {path}")
    return img


def profile_execution(image, kernel, func, label: str):
    """Замеряет время и потребление RAM для переданной функции."""
    tracemalloc.start()
    start_time = time.perf_counter()

    result = func(image, kernel)

    end_time = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    exec_time = end_time - start_time
    peak_mem_mb = peak_mem / (1024 * 1024)

    print(f"[{label}] Время: {exec_time:.4f} сек | Пик RAM: {peak_mem_mb:.2f} MB")
    return result


def process_and_save(image: np.ndarray, kernel_name: str, kernel: np.ndarray):

    # Запуск с профилированием
    naive_result = profile_execution(image, kernel, convolve2d_naive, "NAIVE")
    opencv_result = profile_execution(image, kernel, convolve2d_opencv, "OPENCV")

    naive_result = convolve2d_naive(image, kernel)
    opencv_result = convolve2d_opencv(image, kernel)

    naive_uint8 = clip_to_uint8(naive_result)
    opencv_uint8 = clip_to_uint8(opencv_result)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, f"{kernel_name}_naive.png"), naive_uint8)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, f"{kernel_name}_opencv.png"), opencv_uint8)

    # заменила неиспользуемую переменную fig на нижнее подчеркивание
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(naive_uint8, cmap="gray")
    axes[1].set_title(f"Naive ({kernel_name})")
    axes[1].axis("off")

    axes[2].imshow(opencv_uint8, cmap="gray")
    axes[2].set_title(f"OpenCV ({kernel_name})")
    axes[2].axis("off")

    plt.suptitle(f"Kernel: {kernel_name}", fontsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{kernel_name}_comparison.png"), dpi=150)
    plt.show()


def main():
    image = load_grayscale(INPUT_PATH)
    print(f"Загружено изображение: {image.shape}")

    for name, kernel in KERNELS.items():
        print(f"\nОбработка ядром: {name}")
        process_and_save(image, name, kernel)

    print(f"\nРезультаты сохранены в: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
