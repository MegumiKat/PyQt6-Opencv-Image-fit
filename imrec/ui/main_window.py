"""主窗口：四宫格预览、热力图参数、识别管线。"""

from __future__ import annotations

import html

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from imrec.config import IMG_CELL_MIN, PREVIEW_MAX_SIDE
from imrec.i18n import UI_STR
from imrec.qt_bridge import html_bold_before_colon, numpy_bgr_to_qpixmap
from imrec.ui.dialogs import ImageZoomDialog
from imrec.ui.widgets import ClickableLabel
from imrec.vision.binary import adaptive_binary_mask, mask_objects_as_foreground
from imrec.vision.heatmap import draw_grid_heatmap, grid_circle_counts
from imrec.vision.watershed import local_watershed_circles, watershed_split_instances


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._lang = "en"
        self.resize(1500, 820)

        self._bgr: np.ndarray | None = None
        self._gray: np.ndarray | None = None
        self._last_vis_fit: np.ndarray | None = None
        self._last_detected: list[tuple[float, float, float]] | None = None
        self._last_status_core: str | None = None
        self._last_result_array: np.ndarray | None = None
        self._last_heat: np.ndarray | None = None

        self._open_btn = QPushButton()
        self._open_btn.clicked.connect(self._open_image)

        self._spin_m = QSpinBox()
        self._spin_m.setRange(1, 200)
        self._spin_m.setValue(6)
        self._spin_m.valueChanged.connect(self._on_heatmap_params_changed)
        self._spin_n = QSpinBox()
        self._spin_n.setRange(1, 200)
        self._spin_n.setValue(8)
        self._spin_n.valueChanged.connect(self._on_heatmap_params_changed)

        self._radio_adapt = QRadioButton()
        self._radio_otsu = QRadioButton()
        self._radio_hough = QRadioButton()
        self._radio_adapt.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self._radio_adapt)
        mode_group.addButton(self._radio_otsu)
        mode_group.addButton(self._radio_hough)
        mode_group.buttonClicked.connect(self._on_mode_changed)

        self._chk_clahe = QCheckBox()
        self._chk_clahe.setChecked(False)
        self._chk_clahe.toggled.connect(self._on_clahe_toggled)

        self._chk_watershed = QCheckBox()
        self._chk_watershed.setChecked(False)
        self._chk_watershed.toggled.connect(self._on_overlay_style_changed)

        self._chk_black_circles_only = QCheckBox()
        self._chk_black_circles_only.setChecked(False)
        self._chk_black_circles_only.toggled.connect(self._on_overlay_style_changed)

        cw, ch = IMG_CELL_MIN
        self._orig_label = ClickableLabel()
        self._result_label = ClickableLabel()
        for lb in (self._orig_label, self._result_label):
            lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lb.setMinimumSize(cw, ch)
            lb.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Ignored,
            )
            lb.setStyleSheet("border: 1px solid palette(mid);")

        self._result_caption = QLabel()

        self._fit_label = ClickableLabel()
        self._fit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fit_label.setMinimumSize(cw, ch)
        self._fit_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self._fit_label.setStyleSheet("border: 1px solid palette(mid);")

        self._heat_label = ClickableLabel()
        self._heat_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._heat_label.setMinimumSize(cw, ch)
        self._heat_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self._heat_label.setStyleSheet("border: 1px solid palette(mid);")

        self._orig_label.clicked.connect(
            lambda: self._zoom_image(self._t("zoom_orig"), self._bgr)
        )
        self._result_label.clicked.connect(
            lambda: self._zoom_image(self._t("zoom_out"), self._last_result_array)
        )
        self._fit_label.clicked.connect(
            lambda: self._zoom_image(self._t("zoom_fit"), self._last_vis_fit)
        )
        self._heat_label.clicked.connect(
            lambda: self._zoom_image(self._t("zoom_heat"), self._last_heat)
        )

        self._status_label = QLabel()
        self._status_label.setTextFormat(Qt.TextFormat.PlainText)
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self._status_label.setMinimumHeight(88)

        self._lab_params = QLabel()
        self._lab_params.setStyleSheet("font-weight: bold;")
        row_params = QHBoxLayout()
        row_params.addWidget(self._open_btn)
        row_params.addSpacing(20)
        self._lbl_m = QLabel()
        row_params.addWidget(self._lbl_m)
        row_params.addWidget(self._spin_m)
        row_params.addSpacing(16)
        self._lbl_n = QLabel()
        row_params.addWidget(self._lbl_n)
        row_params.addWidget(self._spin_n)
        row_params.addStretch()
        self._lang_btn = QPushButton()
        self._lang_btn.clicked.connect(self._toggle_language)
        row_params.addWidget(self._lang_btn)

        self._lab_choices = QLabel()
        self._lab_choices.setStyleSheet("font-weight: bold;")

        row_mode = QHBoxLayout()
        row_mode.addWidget(self._radio_adapt)
        row_mode.addWidget(self._radio_otsu)
        row_mode.addWidget(self._radio_hough)
        row_mode.addStretch()

        self._lab_result = QLabel()
        self._lab_result.setStyleSheet("font-weight: bold;")

        # 隐藏模式/勾选/识别结果文案区，逻辑仍使用上述控件的当前值
        self._advanced_options_panel = QWidget()
        lo_adv = QVBoxLayout(self._advanced_options_panel)
        lo_adv.setContentsMargins(0, 0, 0, 0)
        lo_adv.addWidget(self._lab_choices)
        lo_adv.addLayout(row_mode)
        lo_adv.addWidget(self._chk_clahe)
        lo_adv.addWidget(self._chk_watershed)
        lo_adv.addWidget(self._chk_black_circles_only)
        lo_adv.addSpacing(8)
        lo_adv.addWidget(self._lab_result)
        lo_adv.addWidget(self._status_label)
        self._advanced_options_panel.setVisible(False)

        scroll_orig = QScrollArea()
        scroll_orig.setWidgetResizable(True)
        scroll_orig.setWidget(self._orig_label)
        scroll_orig.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        scroll_orig.setMinimumHeight(ch + 40)

        scroll_result = QScrollArea()
        scroll_result.setWidgetResizable(True)
        scroll_result.setWidget(self._result_label)
        scroll_result.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        scroll_result.setMinimumHeight(ch + 40)

        scroll_fit = QScrollArea()
        scroll_fit.setWidgetResizable(True)
        scroll_fit.setWidget(self._fit_label)
        scroll_fit.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        scroll_fit.setMinimumHeight(ch + 40)

        scroll_heat = QScrollArea()
        scroll_heat.setWidgetResizable(True)
        scroll_heat.setWidget(self._heat_label)
        scroll_heat.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        scroll_heat.setMinimumHeight(ch + 40)

        self._title_orig = QLabel()
        self._title_orig.setStyleSheet("font-weight: bold;")
        cell_orig = QWidget()
        lo = QVBoxLayout(cell_orig)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.addWidget(self._title_orig)
        lo.addWidget(scroll_orig, 1)

        cell_result = QWidget()
        lr = QVBoxLayout(cell_result)
        lr.setContentsMargins(0, 0, 0, 0)
        lr.addWidget(self._result_caption)
        lr.addWidget(scroll_result, 1)

        self._title_fit = QLabel()
        self._title_fit.setStyleSheet("font-weight: bold;")
        cell_fit = QWidget()
        lf = QVBoxLayout(cell_fit)
        lf.setContentsMargins(0, 0, 0, 0)
        lf.addWidget(self._title_fit)
        lf.addWidget(scroll_fit, 1)

        self._title_heat = QLabel()
        self._title_heat.setStyleSheet("font-weight: bold;")
        cell_heat = QWidget()
        lh = QVBoxLayout(cell_heat)
        lh.setContentsMargins(0, 0, 0, 0)
        lh.addWidget(self._title_heat)
        lh.addWidget(scroll_heat, 1)

        images_grid = QGridLayout()
        images_grid.setSpacing(10)
        images_grid.setRowStretch(0, 1)
        images_grid.setRowStretch(1, 1)
        images_grid.setColumnStretch(0, 1)
        images_grid.setColumnStretch(1, 1)
        images_grid.addWidget(cell_orig, 0, 0)
        images_grid.addWidget(cell_result, 0, 1)
        images_grid.addWidget(cell_fit, 1, 0)
        images_grid.addWidget(cell_heat, 1, 1)

        root = QVBoxLayout()
        root.addWidget(self._lab_params)
        root.addLayout(row_params)
        root.addSpacing(8)
        root.addWidget(self._advanced_options_panel)
        root.addLayout(images_grid, 1)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self._apply_ui_language()

    def _t(self, key: str, **kw: object) -> str:
        s = UI_STR[self._lang][key]
        return s.format(**kw) if kw else s

    def _apply_ui_language(self) -> None:
        self.setWindowTitle(self._t("win_title"))
        self._lang_btn.setText(self._t("lang_switch"))
        self._open_btn.setText(self._t("open_image"))
        self._lab_params.setText(self._t("param_input"))
        self._lbl_m.setText(self._t("lab_m"))
        self._lbl_n.setText(self._t("lab_n"))
        self._lab_choices.setText(self._t("options"))
        self._lab_result.setText(self._t("rec_title"))
        self._radio_adapt.setText(self._t("radio_adapt"))
        self._radio_otsu.setText(self._t("radio_otsu"))
        self._radio_hough.setText(self._t("radio_hough"))
        self._chk_clahe.setText(self._t("chk_clahe"))
        self._chk_watershed.setText(self._t("chk_watershed"))
        self._chk_black_circles_only.setText(self._t("chk_black_only"))
        self._title_orig.setText(self._t("orig"))
        self._title_fit.setText(self._t("title_fit"))
        self._title_heat.setText(self._t("heat"))
        tip = self._t("zoom_tip")
        for lb in (
            self._orig_label,
            self._result_label,
            self._fit_label,
            self._heat_label,
        ):
            lb.setToolTip(tip)
        if self._gray is None:
            self._status_label.setTextFormat(Qt.TextFormat.PlainText)
            self._status_label.setText(self._t("please_open"))
            self._orig_label.setText(self._t("orig"))
            self._result_label.setText(self._t("proc_result"))
            self._result_caption.setText(self._t("output_default"))
            self._fit_label.setText(self._t("fit_preview"))
            self._heat_label.setText(self._t("heat"))

    def _toggle_language(self) -> None:
        self._lang = "en" if self._lang == "zh" else "zh"
        self._apply_ui_language()
        if self._bgr is not None and self._gray is not None:
            self._refresh_result_view()
            self._contour_and_fit()

    def _heatmap_result_note(
        self,
        m: int,
        n: int,
        has_circles: bool,
        cmin: int,
        cmax: int,
    ) -> str:
        if self._lang == "zh":
            if has_circles:
                return (
                    f"热力图 {m}×{n}：各格覆盖圆数范围 [{cmin}, {cmax}]；"
                    f"色标按本图自动：最少→蓝、最多→红。"
                )
            return (
                f"热力图 {m}×{n}：当前无圆，各格计数为 0；色标为单色蓝。"
            )
        if has_circles:
            return (
                f"Heatmap {m}×{n}: per-cell hit counts in [{cmin}, {cmax}]; "
                f"color scale: min→blue, max→red."
            )
        return (
            f"Heatmap {m}×{n}: no circles; all cells 0; solid blue."
        )

    def _fmt_binary_summary(
        self,
        mode_name: str,
        src: str,
        clahe_note: str,
        inv_note: str,
        mode_note: str,
        n_contour: int,
        n_circle: int,
        ell_txt: str,
        circ_label: str,
    ) -> str:
        if self._lang == "zh":
            return (
                f"当前：{mode_name}。依据：{src}{clahe_note}{inv_note}。 {mode_note}"
                f"轮廓段数：{n_contour}；{circ_label}：{n_circle}；{ell_txt}。"
            )
        return (
            f"Mode: {mode_name}. Basis: {src}{clahe_note}{inv_note}. {mode_note}"
            f"Contour segments: {n_contour}; {circ_label}: {n_circle}; {ell_txt}."
        )

    def _set_result_display(self, detail: str, heat_note: str = "") -> None:
        self._status_label.setTextFormat(Qt.TextFormat.RichText)
        parts: list[str] = []
        d = html_bold_before_colon(detail)
        if d:
            parts.append(d)
        h = html_bold_before_colon(heat_note)
        if h:
            parts.append(h)
        empty = html.escape(self._t("no_output"), quote=False)
        self._status_label.setText(
            "<br/><br/>".join(parts) if parts else empty
        )

    def _zoom_image(self, title: str, arr: np.ndarray | None) -> None:
        if arr is None or arr.size == 0:
            return
        ImageZoomDialog(self, title, arr).exec()

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._t("dlg_pick_image"),
            "",
            self._t("dlg_filter"),
        )
        if not path:
            return

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            QMessageBox.warning(
                self,
                self._t("err_read_title"),
                self._t("err_read_body", path=path),
            )
            return

        self._gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self._bgr = bgr
        self._last_vis_fit = None
        self._last_detected = None
        self._last_status_core = None
        self._last_heat = None
        self._fit_label.clear()
        self._fit_label.setText(self._t("fit_preview"))
        self._heat_label.clear()
        self._heat_label.setText(self._t("heat"))
        self._status_label.setTextFormat(Qt.TextFormat.PlainText)
        self._status_label.setText(self._t("loading"))

        self._orig_label.setPixmap(
            numpy_bgr_to_qpixmap(bgr, max_side=PREVIEW_MAX_SIDE)
        )
        self._refresh_result_view()
        self._contour_and_fit()

    def _on_mode_changed(self) -> None:
        self._refresh_result_view()
        if self._bgr is not None and self._gray is not None:
            self._contour_and_fit()

    def _on_clahe_toggled(self) -> None:
        if self._gray is not None:
            self._refresh_result_view()
            if self._bgr is not None:
                self._contour_and_fit()

    def _on_overlay_style_changed(self) -> None:
        if self._gray is not None and self._bgr is not None:
            self._contour_and_fit()

    def _on_heatmap_params_changed(self) -> None:
        if (
            self._last_vis_fit is not None
            and self._last_status_core is not None
            and self._last_detected is not None
        ):
            self._apply_heatmap_panel(
                self._last_vis_fit,
                self._last_detected,
                self._last_status_core,
            )

    def _base_gray(self) -> np.ndarray:
        assert self._gray is not None
        if self._chk_clahe.isChecked():
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(self._gray)
        return self._gray

    def _is_hough_mode(self) -> bool:
        return self._radio_hough.isChecked()

    def _refresh_result_view(self) -> None:
        if self._gray is None:
            self._result_label.clear()
            self._result_label.setText(self._t("proc_result"))
            self._result_caption.setText(self._t("output_default"))
            self._last_result_array = None
            return

        g = self._base_gray()

        if self._radio_otsu.isChecked():
            _, bin_otsu = cv2.threshold(
                g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            self._last_result_array = bin_otsu
            self._result_caption.setText(self._t("caption_otsu"))
            self._result_label.setPixmap(
                numpy_bgr_to_qpixmap(bin_otsu, max_side=PREVIEW_MAX_SIDE)
            )
        elif self._radio_adapt.isChecked():
            adapt = adaptive_binary_mask(g)
            self._last_result_array = adapt
            self._result_caption.setText(self._t("caption_adapt"))
            self._result_label.setPixmap(
                numpy_bgr_to_qpixmap(adapt, max_side=PREVIEW_MAX_SIDE)
            )
        else:
            blur = cv2.GaussianBlur(g, (9, 9), 0)
            self._last_result_array = blur
            self._result_caption.setText(self._t("caption_blur"))
            self._result_label.setPixmap(
                numpy_bgr_to_qpixmap(blur, max_side=PREVIEW_MAX_SIDE)
            )

    def _mask_for_contours(self) -> np.ndarray:
        assert self._gray is not None
        g = self._base_gray()
        if self._radio_otsu.isChecked():
            _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return m
        return adaptive_binary_mask(g)

    def _hough_blur(self, g: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(g, (9, 9), 0)

    def _hough_circles_primary(self, blur: np.ndarray) -> np.ndarray | None:
        rows, cols = blur.shape[:2]
        side = min(rows, cols)
        min_r = max(4, side // 120)
        max_r = side // 10
        min_dist = max(min_r * 2, side // 55)
        return cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=float(min_dist),
            param1=100,
            param2=28,
            minRadius=min_r,
            maxRadius=max_r,
        )

    def _hough_secondary_large_circles(
        self, blur: np.ndarray, circles: np.ndarray
    ) -> list[tuple[float, float, float]]:
        """方案A（灰度）：半径明显大于中位数的圆在 ROI 内二次 Hough。"""
        rows, cols = blur.shape[:2]
        pts = circles[0]
        rs = pts[:, 2]
        med = float(np.median(rs))
        extra: list[tuple[float, float, float]] = []
        for cx, cy, r in pts:
            if r < max(2.3 * med, 14.0):
                continue
            pad = int(min(r * 2.0, min(rows, cols) * 0.25))
            x0, y0 = max(0, int(cx) - pad), max(0, int(cy) - pad)
            x1, y1 = min(cols, int(cx) + pad), min(rows, int(cy) + pad)
            roi = blur[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            sub = cv2.HoughCircles(
                roi,
                cv2.HOUGH_GRADIENT,
                dp=1.1,
                minDist=float(max(8, r * 0.45)),
                param1=100,
                param2=22,
                minRadius=max(3, int(r * 0.22)),
                maxRadius=int(min(r * 0.92, min(roi.shape) // 2)),
            )
            if sub is None:
                continue
            for sx, sy, sr in sub[0]:
                gx, gy = sx + x0, sy + y0
                if (gx - cx) ** 2 + (gy - cy) ** 2 < (r * 0.35) ** 2:
                    continue
                extra.append((float(gx), float(gy), float(sr)))
        return extra

    def _dedupe_circles(
        self,
        circles: list[tuple[float, float, float]],
        min_center_dist: float,
    ) -> list[tuple[float, float, float]]:
        if not circles:
            return []
        circles = sorted(circles, key=lambda t: -t[2])
        kept: list[tuple[float, float, float]] = []
        for c in circles:
            cx, cy, r = c
            ok = True
            for kx, ky, kr in kept:
                d = np.hypot(cx - kx, cy - ky)
                if d < min_center_dist or d < (r + kr) * 0.35:
                    ok = False
                    break
            if ok:
                kept.append(c)
        return kept

    def _apply_heatmap_panel(
        self,
        vis: np.ndarray,
        detected: list[tuple[float, float, float]],
        status_text: str,
    ) -> None:
        self._last_vis_fit = vis.copy()
        self._last_detected = list(detected)
        self._last_status_core = status_text
        m, n = self._spin_m.value(), self._spin_n.value()
        hh, ww = vis.shape[:2]
        heat = draw_grid_heatmap(hh, ww, detected, m, n)
        self._last_heat = heat
        self._fit_label.setPixmap(
            numpy_bgr_to_qpixmap(vis, max_side=PREVIEW_MAX_SIDE)
        )
        self._heat_label.setPixmap(
            numpy_bgr_to_qpixmap(heat, max_side=PREVIEW_MAX_SIDE)
        )
        if detected:
            hc, wc = vis.shape[:2]
            cnt = grid_circle_counts(hc, wc, m, n, detected)
            heat_note = self._heatmap_result_note(
                m,
                n,
                True,
                int(cnt.min()),
                int(cnt.max()),
            )
        else:
            heat_note = self._heatmap_result_note(m, n, False, 0, 0)
        self._set_result_display(status_text, heat_note)

    def _contour_and_fit(self) -> None:
        if self._bgr is None or self._gray is None:
            return

        vis = self._bgr.copy()
        circles_only = self._chk_black_circles_only.isChecked()
        circle_bgr = (0, 0, 0) if circles_only else (0, 255, 0)

        if self._is_hough_mode():
            g = self._base_gray()
            blur = self._hough_blur(g)
            raw = self._hough_circles_primary(blur)
            n_circle = 0
            all_circles: list[tuple[float, float, float]] = []
            if raw is not None:
                for x, y, r in raw[0]:
                    all_circles.append((float(x), float(y), float(r)))
                all_circles.extend(self._hough_secondary_large_circles(blur, raw))
            min_d = max(8.0, min(blur.shape) / 80.0)
            all_circles = self._dedupe_circles(all_circles, min_d)
            circle_thick = 2 if circles_only else 2
            for cx, cy, r in all_circles:
                ir = int(max(1, round(r)))
                cv2.circle(
                    vis,
                    (int(round(cx)), int(round(cy))),
                    ir,
                    circle_bgr,
                    circle_thick,
                )
                n_circle += 1
            clahe_note = (
                self._t("hough_clahe_yes") if self._chk_clahe.isChecked() else ""
            )
            st = self._t("hough_status", clahe=clahe_note, n=n_circle)
            self._apply_heatmap_panel(vis, all_circles, st)
            return

        raw = self._mask_for_contours()
        mask, auto_invert = mask_objects_as_foreground(raw)
        h, w = mask.shape[:2]
        min_area = max(64, (w * h) // 3000)
        min_cell_area = max(30, (w * h) // 12000)

        n_contour = 0
        n_circle = 0
        n_ellipse = 0
        detected: list[tuple[float, float, float]] = []

        if self._chk_watershed.isChecked():
            circle_thick = 2 if circles_only else 1
            ws_img = self._bgr.copy()
            markers = watershed_split_instances(mask, ws_img)
            n_labels = int(markers.max())
            for lbl in range(2, n_labels + 1):
                cell_bin = np.uint8((markers == lbl) * 255)
                if cv2.countNonZero(cell_bin) < min_cell_area:
                    continue
                cnts, _ = cv2.findContours(
                    cell_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for c in cnts:
                    area = float(cv2.contourArea(c))
                    if area < min_cell_area:
                        continue
                    peri = cv2.arcLength(c, True)
                    if peri < 1e-6:
                        continue
                    if not circles_only:
                        cv2.drawContours(vis, [c], -1, (255, 64, 0), 1)
                    (cx, cy), r = cv2.minEnclosingCircle(c)
                    detected.append((float(cx), float(cy), float(r)))
                    center = (int(round(cx)), int(round(cy)))
                    ir = int(max(1, round(r)))
                    cv2.circle(vis, center, ir, circle_bgr, circle_thick)
                    n_contour += 1
                    n_circle += 1
            mode_note = self._t("ws_note", mca=min_cell_area, nc=n_contour)
        else:
            circle_thick = 2
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            areas = [
                float(cv2.contourArea(c))
                for c in contours
                if cv2.contourArea(c) >= min_area
            ]
            a_med = float(np.median(areas)) if areas else float(min_area * 2)
            a_max = max(int(2.5 * a_med), int(3 * min_area))

            for c in contours:
                area = float(cv2.contourArea(c))
                if area < min_area:
                    continue
                peri = cv2.arcLength(c, True)
                if peri < 1e-6:
                    continue

                use_local = area > a_max
                if use_local:
                    layer = np.zeros_like(mask)
                    cv2.drawContours(layer, [c], -1, 255, -1)
                    x, y, ww, hh = cv2.boundingRect(c)
                    pad = max(8, int(0.12 * max(ww, hh)))
                    x0, y0 = max(0, x - pad), max(0, y - pad)
                    x1, y1 = min(w, x + ww + pad), min(h, y + hh + pad)
                    roi_mask = layer[y0:y1, x0:x1]
                    roi_bgr = self._bgr[y0:y1, x0:x1].copy()
                    local = local_watershed_circles(roi_mask, roi_bgr)
                    if len(local) >= 2:
                        for lx, ly, lr in local:
                            gx, gy, gr = lx + x0, ly + y0, lr
                            detected.append((float(gx), float(gy), float(gr)))
                            if not circles_only:
                                cv2.circle(
                                    vis,
                                    (int(round(gx)), int(round(gy))),
                                    int(max(1, round(gr))),
                                    (255, 64, 0),
                                    1,
                                )
                            cv2.circle(
                                vis,
                                (int(round(gx)), int(round(gy))),
                                int(max(1, round(gr))),
                                circle_bgr,
                                circle_thick,
                            )
                            n_contour += 1
                            n_circle += 1
                        continue

                n_contour += 1
                if not circles_only:
                    cv2.drawContours(vis, [c], -1, (255, 64, 0), 1)

                (cx, cy), r = cv2.minEnclosingCircle(c)
                detected.append((float(cx), float(cy), float(r)))
                center = (int(round(cx)), int(round(cy)))
                ir = int(round(r))
                cv2.circle(vis, center, ir, circle_bgr, circle_thick)
                n_circle += 1

                if (not circles_only) and len(c) >= 5:
                    try:
                        ell = cv2.fitEllipse(c)
                        cv2.ellipse(vis, ell, (0, 200, 255), 2)
                        n_ellipse += 1
                    except cv2.error:
                        pass

            mode_note = self._t("sa_note", med=int(round(a_med)), mx=a_max)

        otsu = self._radio_otsu.isChecked()
        mode_name = self._t("mode_otsu") if otsu else self._t("mode_adapt")
        src = self._t("src_otsu") if otsu else self._t("src_adapt")
        clahe_note = (
            self._t("clahe_gray_yes") if self._chk_clahe.isChecked() else ""
        )
        inv_note = self._t("mask_inverted_yes") if auto_invert else ""
        if self._chk_watershed.isChecked():
            ell_txt = self._t("ell_ws")
        elif not circles_only:
            ell_txt = self._t("ell_n", n=n_ellipse)
        else:
            ell_txt = self._t("ell_only")
        circ_label = (
            self._t("circ_black") if circles_only else self._t("circ_green")
        )
        st = self._fmt_binary_summary(
            mode_name,
            src,
            clahe_note,
            inv_note,
            mode_note,
            n_contour,
            n_circle,
            ell_txt,
            circ_label,
        )
        self._apply_heatmap_panel(vis, detected, st)

