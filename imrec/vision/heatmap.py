"""m×n 格与圆相交计数及热力图绘制。"""

from __future__ import annotations

import cv2
import numpy as np

from imrec.config import HEATMAP_BGR_BLUE, HEATMAP_BGR_RED


def circle_intersects_cell(
    cx: float,
    cy: float,
    r: float,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> bool:
    """圆盘是否与轴对齐矩形 [x0,x1)×[y0,y1) 相交（半开区间右/下边界）。"""
    if r <= 0:
        return False
    xi = max(x0, min(cx, x1 - 1e-6))
    yi = max(y0, min(cy, y1 - 1e-6))
    return (xi - cx) ** 2 + (yi - cy) ** 2 <= r * r + 1e-6


def grid_circle_counts(
    h: int,
    w: int,
    m: int,
    n: int,
    circles: list[tuple[float, float, float]],
) -> np.ndarray:
    counts = np.zeros((m, n), dtype=np.int32)
    for i in range(m):
        y0 = int(round(i * h / m))
        y1 = int(round((i + 1) * h / m)) if i < m - 1 else h
        for j in range(n):
            x0 = int(round(j * w / n))
            x1 = int(round((j + 1) * w / n)) if j < n - 1 else w
            for cx, cy, rad in circles:
                if circle_intersects_cell(cx, cy, rad, x0, y0, x1, y1):
                    counts[i, j] += 1
    return counts


def heatmap_blue_red_bgr(cnt: int, cmin: int, cmax: int) -> tuple[int, int, int]:
    """BGR：格内圆数少=蓝，多=红（按本图各格 min～max 线性插值）。"""
    b0, g0, r0 = HEATMAP_BGR_BLUE
    b1, g1, r1 = HEATMAP_BGR_RED
    if cmax == cmin:
        return (b0, g0, r0)
    t = (int(cnt) - cmin) / (cmax - cmin)
    t = max(0.0, min(1.0, t))
    b = int(round(b0 + t * (b1 - b0)))
    g = int(round(g0 + t * (g1 - g0)))
    r = int(round(r0 + t * (r1 - r0)))
    return (b, g, r)


def heatmap_bar_bgr_at_t(t: float) -> tuple[int, int, int]:
    """t∈[0,1]：0=蓝（低计数），1=红（高计数）。"""
    t = max(0.0, min(1.0, t))
    b0, g0, r0 = HEATMAP_BGR_BLUE
    b1, g1, r1 = HEATMAP_BGR_RED
    return (
        int(round(b0 + t * (b1 - b0))),
        int(round(g0 + t * (g1 - g0))),
        int(round(r0 + t * (r1 - r0))),
    )


def heatmap_tick_values_range(cmin: int, cmax: int) -> list[int]:
    """在 [cmin, cmax] 上取刻度，疏密随跨度调整。"""
    lo, hi = int(cmin), int(cmax)
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        return [lo]
    span = hi - lo
    if span <= 12:
        return list(range(lo, hi + 1))
    step = max(1, round(span / 6))
    nice = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000]
    use_step = next((s for s in nice if s >= step), step)
    vals = list(range(lo, hi + 1, use_step))
    if vals[-1] != hi:
        vals.append(hi)
    if vals[0] != lo:
        vals.insert(0, lo)
    return sorted(set(vals))


def draw_grid_heatmap(
    h: int,
    w: int,
    circles: list[tuple[float, float, float]],
    m: int,
    n: int,
) -> np.ndarray:
    """
    格子纯色热力图；色标为各格圆数 min～max（蓝→红）。
    右侧竖条与数字刻度同映射；刻度字较大。
    """
    counts = grid_circle_counts(h, w, m, n, circles)
    cmin = int(counts.min())
    cmax = int(counts.max())

    pad = 12
    bar_w = 34
    tick_w = 64 if max(abs(cmin), abs(cmax)) >= 100 else 52
    out_h = h
    out_w = w + pad + bar_w + tick_w
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for i in range(m):
        y0 = int(round(i * h / m))
        y1 = int(round((i + 1) * h / m)) if i < m - 1 else h
        for j in range(n):
            x0 = int(round(j * w / n))
            x1 = int(round((j + 1) * w / n)) if j < n - 1 else w
            cnt = int(counts[i, j])
            bgr = heatmap_blue_red_bgr(cnt, cmin, cmax)
            cv2.rectangle(
                canvas,
                (x0, y0),
                (max(x0, x1 - 1), max(y0, y1 - 1)),
                bgr,
                -1,
            )

    grid_line = (45, 45, 45)
    for i in range(1, m):
        yy = int(round(i * h / m))
        cv2.line(canvas, (0, yy), (w - 1, yy), grid_line, 1)
    for j in range(1, n):
        xx = int(round(j * w / n))
        cv2.line(canvas, (xx, 0), (xx, h - 1), grid_line, 1)
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (90, 90, 90), 1)

    bx0 = w + pad
    bx1 = bx0 + bar_w
    if h <= 1:
        bgr = heatmap_blue_red_bgr(cmax, cmin, cmax)
        cv2.rectangle(canvas, (bx0, 0), (bx1 - 1, h - 1), bgr, -1)
    else:
        for y in range(h):
            t = 1.0 - y / (h - 1)
            bb, gg, rr = heatmap_bar_bgr_at_t(t)
            cv2.line(canvas, (bx0, y), (bx1 - 1, y), (bb, gg, rr), 1)
    cv2.rectangle(canvas, (bx0, 0), (bx1 - 1, h - 1), (200, 200, 200), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.72
    th = 2
    tick_color = (255, 255, 255)
    tx = bx1 + 6

    lo, hi = cmin, cmax
    tick_vals = heatmap_tick_values_range(cmin, cmax)

    for val in tick_vals:
        if hi == lo:
            y_pos = h // 2
        else:
            y_pos = int(round((h - 1) * (1.0 - (val - lo) / (hi - lo))))
        y_pos = max(14, min(h - 6, y_pos))
        cv2.line(canvas, (bx0 - 4, y_pos), (bx1 + 3, y_pos), tick_color, 1)
        cv2.putText(
            canvas,
            str(val),
            (tx, y_pos + 6),
            font,
            fs,
            tick_color,
            th,
            cv2.LINE_AA,
        )

    return canvas
