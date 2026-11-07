import cv2
import numpy as np


def convolve2d_naive(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="edge")
    h, w = image.shape
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i : i + kh, j : j + kw]
            output[i, j] = np.sum(region * kernel)
    return output


def convolve2d_opencv(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


def clip_to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)
