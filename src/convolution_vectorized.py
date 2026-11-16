import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def convolve2d_vectorized(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Вернуть отклик float32 для 2D-изображения uint8 и нечётного ядра float32.

    Как и в convolve2d_naive, ядро применяется без поворота. Для ядер
    проекта, симметричных при повороте на 180°, это даёт результат свёртки.
    """
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Изображение должно быть непустым двумерным массивом.")
    if kernel.ndim != 2 or any(size % 2 == 0 for size in kernel.shape):
        raise ValueError("Ядро должно быть двумерным с нечётными размерами.")
    if image.dtype != np.uint8 or kernel.dtype != np.float32:
        raise TypeError("Ожидаются изображение uint8 и ядро float32.")

    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    # Копируем изображение в буфер с повторёнными краями, как в наивной версии.
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="edge")

    # Окна (H, W, kh, kw) используют шаги адресации (strides) буфера padded.
    # Перекрывающиеся окна ссылаются на одни и те же пиксели без их копирования.
    windows = sliding_window_view(padded, (kh, kw))

    # NumPy умножает и суммирует по m, n в скомпилированном цикле.
    # Явный массив всех произведений не создаётся. Отклик хранится в float32,
    # чтобы сохранить отрицательные значения и значения выше 255 до клиппинга.
    return np.einsum("ijmn,mn->ij", windows, kernel, dtype=np.float32, optimize=False)
