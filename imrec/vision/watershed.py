"""分水岭分割（整图与 ROI 局部分割）。"""

from __future__ import annotations

import cv2
import numpy as np


def watershed_split_instances(mask_fg: np.ndarray, guide_bgr: np.ndarray) -> np.ndarray:
    """整图分水岭（可选）；峰阈值较激进，易过分割。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dmax = float(dist.max()) if dist.size else 0.0
    thr = 0.42 * dmax if dmax > 0 else 0
    _, sure_fg = cv2.threshold(dist, thr, 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers.astype(np.int32) + 1
    markers[unknown == 255] = 0

    if n_markers <= 1 and cv2.countNonZero(sure_fg) == 0:
        return np.full(mask_fg.shape, 1, dtype=np.int32)

    cv2.watershed(guide_bgr, markers)
    return markers


def local_watershed_circles(
    mask_roi: np.ndarray, guide_bgr_roi: np.ndarray
) -> list[tuple[float, float, float]]:
    """方案 A：仅在 ROI 内分水岭，峰阈值更高并先腐蚀，减少过切。返回 ROI 坐标系下的 (cx, cy, r)。"""
    if mask_roi.size == 0 or cv2.countNonZero(mask_roi) < 40:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel, iterations=1)
    opening = cv2.erode(opening, kernel, iterations=1)
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dmax = float(dist.max()) if dist.size else 0.0
    thr = 0.58 * dmax if dmax > 0 else 0
    _, sure_fg = cv2.threshold(dist, thr, 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers.astype(np.int32) + 1
    markers[unknown == 255] = 0

    if n_markers <= 1 and cv2.countNonZero(sure_fg) == 0:
        return []

    g = guide_bgr_roi.copy()
    cv2.watershed(g, markers)

    min_a = max(25, mask_roi.size // 400)
    out: list[tuple[float, float, float]] = []
    max_lbl = int(markers.max())
    for lbl in range(2, max_lbl + 1):
        cell = np.uint8((markers == lbl) * 255)
        cnts, _ = cv2.findContours(
            cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in cnts:
            if cv2.contourArea(c) < min_a:
                continue
            (cx, cy), rad = cv2.minEnclosingCircle(c)
            if rad >= 2:
                out.append((cx, cy, rad))
    return out
