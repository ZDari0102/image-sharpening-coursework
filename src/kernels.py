import numpy as np

LAPLACIAN_3x3 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

HIGH_PASS_3x3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)

SHARPEN_3x3 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

KERNELS = {
    "laplacian": LAPLACIAN_3x3,
    "high_pass": HIGH_PASS_3x3,
    "sharpen": SHARPEN_3x3,
}
