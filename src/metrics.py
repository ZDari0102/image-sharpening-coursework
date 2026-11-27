"""Метрики полутоновых изображений uint8 в диапазоне от 0 до 255."""

from typing import cast

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_metrics(original: np.ndarray, filtered: np.ndarray) -> dict[str, float]:
    """Вычислить VoL результата и его PSNR и SSIM относительно original.

    SSIM использует равномерное окно 7×7. Лапласиан вычисляется с ksize=1
    и отражением BORDER_REFLECT_101, дисперсия — с делителем N (ddof=0).
    """
    if original.ndim != 2 or filtered.ndim != 2:
        raise ValueError("Метрики рассчитаны на двумерные полутоновые изображения.")
    if original.shape != filtered.shape:
        raise ValueError("Изображения для сравнения должны иметь одинаковый размер.")
    if original.dtype != np.uint8 or filtered.dtype != np.uint8:
        raise TypeError("Перед расчётом метрик изображения должны иметь тип uint8.")
    if min(original.shape) < 7:
        raise ValueError(
            "Для SSIM с окном 7×7 размеры изображения должны быть не меньше 7×7."
        )

    # Знаковый буфер float64 сохраняет отрицательные значения второй производной.
    # Он выделяется отдельно от выходного буфера алгоритма фильтрации.
    variance_laplacian = float(
        cv2.Laplacian(
            filtered, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_REFLECT_101
        ).var()
    )

    # При нулевой ошибке PSNR равен бесконечности. Проверка также позволяет
    # пропустить выделение рабочих массивов SSIM для строки исходного изображения.
    if np.array_equal(original, filtered):
        psnr_db = float("inf")
        ssim = 1.0
    else:
        psnr_db = float(peak_signal_noise_ratio(original, filtered, data_range=255))
        ssim_score = structural_similarity(
            original,
            filtered,
            data_range=255,
            win_size=7,
            gradient=False,
            full=False,
        )
        # Эти флаги выбирают скалярный результат. Уточняем тип для Pylance,
        # затем приводим скаляр NumPy к float Python для записи в CSV.
        ssim = float(cast(np.float64, ssim_score))

    return {
        "variance_laplacian": variance_laplacian,
        "psnr_db": psnr_db,
        "ssim": ssim,
    }
