"""二值掩膜与极性校正。"""

from __future__ import annotations

import cv2
import numpy as np


def mask_objects_as_foreground(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    white = int(cv2.countNonZero(mask))
    if white * 2 > mask.size:
        return cv2.bitwise_not(mask), True
    return mask, False


def adaptive_binary_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = blur.shape[:2]
    side = min(h, w)
    block = min(31, max(3, (side // 16) | 1))
    if block % 2 == 0:
        block += 1
    return cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        5,
    )
