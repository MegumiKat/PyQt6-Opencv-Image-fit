"""应用入口。"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from imrec.ui.main_window import MainWindow


def run() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
