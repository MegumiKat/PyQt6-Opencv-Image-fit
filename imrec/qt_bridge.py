"""NumPy/OpenCV 与 Qt 之间的转换及富文本小工具。"""

from __future__ import annotations

import html
import re

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication


def html_bold_before_colon(text: str) -> str:
    """将中文/英文冒号前的短语加粗，按 。； 分段。"""
    if not (text or "").strip():
        return ""

    def esc(s: str) -> str:
        return html.escape(s.strip(), quote=False)

    out: list[str] = []
    for block in text.replace("\r", "").split("\n"):
        block = block.strip()
        if not block:
            continue
        chunks = re.split(r"(?<=[。；])|(?<=[;])|(?<=\.)(?=\s)", block)
        for ch in chunks:
            ch = ch.strip()
            if not ch:
                continue
            if "：" in ch:
                pre, _, post = ch.partition("：")
                out.append(f"<b>{esc(pre)}</b>：{esc(post)}")
            elif ":" in ch:
                pre, _, post = ch.partition(":")
                out.append(f"<b>{esc(pre)}</b>:{esc(post)}")
            else:
                out.append(esc(ch))
    return "<br/>".join(out)


def scale_pixmap(pix: QPixmap, max_side: int) -> QPixmap:
    if max(pix.width(), pix.height()) <= max_side:
        return pix
    return pix.scaled(
        max_side,
        max_side,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def numpy_bgr_to_qpixmap(bgr: np.ndarray, max_side: int = 800) -> QPixmap:
    if bgr.ndim == 2:
        h, w = bgr.shape
        img = np.ascontiguousarray(bgr)
        qimg = QImage(
            img.data,
            w,
            h,
            w,
            QImage.Format.Format_Grayscale8,
        )
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimg = QImage(
            rgb.data,
            w,
            h,
            ch * w,
            QImage.Format.Format_RGB888,
        )
    pix = QPixmap.fromImage(qimg.copy())
    return scale_pixmap(pix, max_side)


def zoom_dialog_max_side() -> int:
    app = QApplication.instance()
    if app is None:
        return 1400
    scr = app.primaryScreen()
    if scr is None:
        return 1400
    g = scr.availableGeometry()
    return max(560, min(g.width() - 100, g.height() - 140))
