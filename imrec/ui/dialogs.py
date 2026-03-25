"""对话框。"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from imrec.qt_bridge import numpy_bgr_to_qpixmap, zoom_dialog_max_side


class ImageZoomDialog(QDialog):
    """按屏幕可用区域上限显示大图，可滚动。"""

    def __init__(
        self,
        parent: QWidget | None,
        title: str,
        image: np.ndarray,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        ms = zoom_dialog_max_side()
        pix = numpy_bgr_to_qpixmap(image, max_side=ms)

        lay = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QLabel()
        inner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.setPixmap(pix)
        scroll.setWidget(inner)
        lay.addWidget(scroll)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

        app = QApplication.instance()
        gw, gh = pix.width() + 72, pix.height() + 120
        if app and app.primaryScreen():
            r = app.primaryScreen().availableGeometry()
            gw = min(r.width() - 64, gw)
            gh = min(r.height() - 80, gh)
        self.resize(max(400, gw), max(320, gh))
