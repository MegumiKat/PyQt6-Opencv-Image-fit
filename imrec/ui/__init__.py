"""PyQt6 界面组件。主窗口请使用 `from imrec.ui.main_window import MainWindow`。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imrec.ui.main_window import MainWindow

__all__ = ["MainWindow"]


def __getattr__(name: str):
    if name == "MainWindow":
        from imrec.ui.main_window import MainWindow as _MainWindow

        return _MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
