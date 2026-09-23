import cv2
import numpy as np


def grayscale_fingerprint(image, size=16):
    if image.size == 0:
        return np.zeros((size, size), dtype=np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32) / 255.0

    mean = float(small.mean())
    std = float(small.std())

    if std > 1e-6:
        small = (small - mean) / std
    else:
        small = small - mean

    return small


def fingerprint_similarity(a, b):
    if a.shape != b.shape or a.size == 0:
        return 0.0

    a_flat = a.reshape(-1).astype(np.float32)
    b_flat = b.reshape(-1).astype(np.float32)

    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    if denom < 1e-8:
        return 0.0

    corr = float(np.dot(a_flat, b_flat) / denom)
    return max(0.0, min(1.0, (corr + 1.0) / 2.0))
