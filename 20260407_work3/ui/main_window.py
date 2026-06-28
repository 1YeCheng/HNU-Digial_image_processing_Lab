# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout,
    QScrollArea, QPushButton, QFileDialog, QMessageBox,
    QInputDialog, QSizePolicy, QFrame,
)
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QSize
from PyQt5.QtWidgets import QGraphicsOpacityEffect

from controller.image_controller import ImageController

# ─────────────────────────────────────────────────────────────────────────── #
# Global QSS
# ─────────────────────────────────────────────────────────────────────────── #
APP_STYLE = """
QMainWindow, QWidget {
    background-color: #FFFFFF;
    color: #333333;
    font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
    font-size: 13px;
}
QWidget#header {
    background-color: #FFFFFF;
    border-bottom: 1px solid #EEEEEE;
}
QLabel#img_label {
    background-color: #FAFAFA;
    border: 1px solid #EEEEEE;
    border-radius: 16px;
    color: #BBBBBB;
    font-size: 13px;
}
QLabel#panel_title {
    color: #AAAAAA;
    font-size: 11px;
    letter-spacing: 1px;
    background: transparent;
}
QLabel#section_title {
    color: #AAAAAA;
    font-size: 10px;
    letter-spacing: 2px;
    background: transparent;
    padding-left: 4px;
}
QScrollArea { border: none; background: transparent; }
QScrollBar:horizontal {
    background: transparent; height: 4px; border-radius: 2px;
}
QScrollBar::handle:horizontal {
    background: rgba(124,58,237,0.25); border-radius: 2px; min-width: 30px;
}
QScrollBar::handle:horizontal:hover { background: rgba(124,58,237,0.5); }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
QPushButton#btn_open {
    background-color: #7C3AED; color: #FFFFFF; border: none;
    border-radius: 16px; padding: 6px 20px; font-weight: 600; font-size: 13px;
}
QPushButton#btn_open:hover { background-color: #6D28D9; }
QPushButton#btn_open:pressed { background-color: #5B21B6; }
QPushButton#btn_save, QPushButton#btn_spectrum, QPushButton#btn_reset, QPushButton#btn_info {
    background-color: #F5F5F5; color: #555555;
    border: 1px solid #E8E8E8; border-radius: 16px;
    padding: 6px 16px; font-size: 13px;
}
QPushButton#btn_save:hover, QPushButton#btn_spectrum:hover,
QPushButton#btn_reset:hover, QPushButton#btn_info:hover {
    background-color: #EBEBEB; color: #333333; border-color: #DDDDDD;
}
QPushButton#btn_save:disabled, QPushButton#btn_spectrum:disabled,
QPushButton#btn_reset:disabled, QPushButton#btn_info:disabled {
    color: #CCCCCC; border-color: #F0F0F0; background-color: #FAFAFA;
}
QStatusBar {
    background-color: #FFFFFF; color: #AAAAAA;
    font-size: 11px; border-top: 1px solid #EEEEEE;
}
"""

CARD_NORMAL = """
    QWidget {{ background-color: {bg}; border: 1px solid {border}; border-radius: 16px; }}
    QLabel {{ background: transparent; border: none; }}
"""
CARD_HOVER = """
    QWidget {{ background-color: {hover_bg}; border: 1px solid {hover_border}; border-radius: 16px; }}
    QLabel {{ background: transparent; border: none; }}
"""
CARD_DISABLED = """
    QWidget {{ background-color: #F8F8F8; border: 1px solid #F0F0F0; border-radius: 16px; }}
    QLabel {{ background: transparent; border: none; color: #CCCCCC; }}
"""

GROUP_TOKENS = {
    "空间变换": {
        "strip_bg": "#F0FFF4", "card_bg": "#DCFCE7", "card_border": "#BBF7D0",
        "hover_bg": "#BBF7D0", "hover_border": "#86EFAC",
    },
    "频率域滤波": {
        "strip_bg": "#FFFBEB", "card_bg": "#FEF3C7", "card_border": "#FDE68A",
        "hover_bg": "#FDE68A", "hover_border": "#FCD34D",
    },
}


# ─────────────────────────────────────────────────────────────────────────── #
# WorkerThread
# ─────────────────────────────────────────────────────────────────────────── #
class WorkerThread(QThread):
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, func):
        super().__init__()
        self._func = func

    def run(self):
        try:
            self.result_ready.emit(self._func())
        except Exception as e:
            self.error_occurred.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────── #
# FeatureCard
# ─────────────────────────────────────────────────────────────────────────── #
class FeatureCard(QWidget):
    clicked = pyqtSignal()

    def __init__(self, icon: str, label: str, tokens: dict, parent=None):
        super().__init__(parent)
        self.setFixedSize(QSize(104, 80))
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self._enabled = True
        self._tokens = tokens

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignCenter)

        icon_lbl = QLabel(icon)
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet("font-size: 22px;")

        text_lbl = QLabel(label)
        text_lbl.setAlignment(Qt.AlignCenter)
        text_lbl.setWordWrap(True)
        text_lbl.setStyleSheet("font-size: 11px; color: #555555;")

        layout.addWidget(icon_lbl)
        layout.addWidget(text_lbl)
        self._apply_normal()

    def _apply_normal(self):
        t = self._tokens
        self.setStyleSheet(CARD_NORMAL.format(bg=t["card_bg"], border=t["card_border"]))

    def _apply_hover(self):
        t = self._tokens
        self.setStyleSheet(CARD_HOVER.format(hover_bg=t["hover_bg"], hover_border=t["hover_border"]))

    def enterEvent(self, e):
        if self._enabled:
            self._apply_hover()
        super().enterEvent(e)

    def leaveEvent(self, e):
        if self._enabled:
            self._apply_normal()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if self._enabled and e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)

    def setEnabled(self, enabled: bool):
        self._enabled = enabled
        self.setCursor(QCursor(Qt.PointingHandCursor if enabled else Qt.ForbiddenCursor))
        if enabled:
            self._apply_normal()
        else:
            self.setStyleSheet(CARD_DISABLED)


# ─────────────────────────────────────────────────────────────────────────── #
# MainWindow
# ─────────────────────────────────────────────────────────────────────────── #
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("数字图像处理 — 傅里叶频谱与滤波演示")
        self.resize(1600, 860)
        self.setMinimumSize(1200, 700)
        self.setStyleSheet(APP_STYLE)

        self.controller = ImageController()
        self._worker = None
        self._image_loaded = False
        self._all_cards: list[FeatureCard] = []
        self._need_image_cards: list[FeatureCard] = []
        # 当前频谱（用于保存）
        self._current_spectrum = None

        self._init_ui()

    # ── UI 构建 ──────────────────────────────────────────────────────────────

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        root.addLayout(self._build_image_stage(), stretch=1)
        root.addWidget(self._build_card_strip())

        self.statusBar().showMessage("就绪  ·  请先打开一张图像")

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("header")
        header.setFixedHeight(52)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(10)

        title = QLabel("数字图像处理演示系统")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #333333; background: transparent;")

        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setStyleSheet("color: #EEEEEE;")

        btn_open = QPushButton("打开图像")
        btn_open.setObjectName("btn_open")
        btn_open.setFixedHeight(34)
        btn_open.clicked.connect(self.open_image)

        self._btn_save = QPushButton("保存结果")
        self._btn_save.setObjectName("btn_save")
        self._btn_save.setFixedHeight(34)
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self.save_result)

        self._btn_spectrum = QPushButton("保存频谱")
        self._btn_spectrum.setObjectName("btn_spectrum")
        self._btn_spectrum.setFixedHeight(34)
        self._btn_spectrum.setEnabled(False)
        self._btn_spectrum.clicked.connect(self.save_spectrum)

        self._btn_reset = QPushButton("恢复原图")
        self._btn_reset.setObjectName("btn_reset")
        self._btn_reset.setFixedHeight(34)
        self._btn_reset.setEnabled(False)
        self._btn_reset.clicked.connect(self.reset)

        self._btn_info = QPushButton("图像信息")
        self._btn_info.setObjectName("btn_info")
        self._btn_info.setFixedHeight(34)
        self._btn_info.setEnabled(False)
        self._btn_info.clicked.connect(self.show_info)

        layout.addWidget(title)
        layout.addWidget(div)
        layout.addWidget(btn_open)
        layout.addWidget(self._btn_save)
        layout.addWidget(self._btn_spectrum)
        layout.addWidget(self._btn_reset)
        layout.addWidget(self._btn_info)
        layout.addStretch()
        return header

    def _build_image_stage(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setContentsMargins(16, 12, 16, 8)
        layout.setSpacing(12)

        left = self._make_panel("原始图像")
        self.label_src = left.findChild(QLabel, "img_label")

        mid = self._make_panel("处理结果")
        self.label_dst = mid.findChild(QLabel, "img_label")

        right = self._make_panel("傅里叶频谱")
        self.label_spectrum = right.findChild(QLabel, "img_label")

        # 淡入动画作用于处理结果面板
        self._opacity_effect = QGraphicsOpacityEffect(self.label_dst)
        self._opacity_effect.setOpacity(1.0)
        self.label_dst.setGraphicsEffect(self._opacity_effect)
        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_anim.setDuration(280)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.OutCubic)

        layout.addWidget(left, 1)
        layout.addWidget(mid, 1)
        layout.addWidget(right, 1)
        return layout

    def _make_panel(self, title_text: str) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel(title_text.upper())
        title.setObjectName("panel_title")
        title.setAlignment(Qt.AlignCenter)

        img_label = QLabel(title_text)
        img_label.setObjectName("img_label")
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(title)
        layout.addWidget(img_label)
        return panel

    def _build_card_strip(self) -> QWidget:
        container = QWidget()
        container.setFixedHeight(220)
        container.setStyleSheet("background-color: #F9FAFB; border-top: 1px solid #EEEEEE;")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(6)

        groups = [
            ("空间变换", [
                ("🔍", "放大 2×",     self.zoom_in,           True),
                ("🔎", "缩小 ½",      self.zoom_out,          True),
                ("↔️", "平移",         self.translate,         True),
                ("🔃", "旋转",         self.rotate,            True),
                ("🌫️", "高斯平滑",    self.gaussian_smooth,   True),
                ("✏️", "拉普拉斯锐化", self.laplacian_sharpen, True),
                ("🧹", "中值去噪",     self.median_denoise,    True),
            ]),
            ("频率域滤波", [
                ("📉", "低通滤波",     self.lowpass_filter,    True),
                ("📈", "高通滤波",     self.highpass_filter,   True),
            ]),
        ]

        for group_name, cards_def in groups:
            tokens = GROUP_TOKENS[group_name]
            row = QWidget()
            row.setStyleSheet(f"background-color: {tokens['strip_bg']}; border-radius: 16px;")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 4, 10, 4)
            row_layout.setSpacing(8)

            sec_lbl = QLabel(group_name.upper())
            sec_lbl.setObjectName("section_title")
            sec_lbl.setFixedWidth(72)
            sec_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            row_layout.addWidget(sec_lbl)

            scroll = QScrollArea()
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidgetResizable(True)
            scroll.setFixedHeight(88)

            inner = QWidget()
            inner.setStyleSheet("background: transparent;")
            inner_layout = QHBoxLayout(inner)
            inner_layout.setContentsMargins(4, 2, 4, 2)
            inner_layout.setSpacing(8)
            inner_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            for icon, label, slot, need_img in cards_def:
                card = FeatureCard(icon, label, tokens)
                card.clicked.connect(slot)
                inner_layout.addWidget(card)
                self._all_cards.append(card)
                if need_img:
                    self._need_image_cards.append(card)
                    card.setEnabled(False)

            inner_layout.addStretch()
            scroll.setWidget(inner)
            row_layout.addWidget(scroll)
            layout.addWidget(row)

        return container

    # ── 工具方法 ─────────────────────────────────────────────────────────────

    def _set_processing_state(self, processing: bool):
        if processing:
            for c in self._all_cards:
                c.setEnabled(False)
            self._btn_save.setEnabled(False)
            self._btn_spectrum.setEnabled(False)
            self._btn_reset.setEnabled(False)
            self._btn_info.setEnabled(False)
            self.statusBar().showMessage("处理中，请稍候…")
        else:
            for c in self._all_cards:
                c.setEnabled(self._image_loaded if c in self._need_image_cards else True)
            if self._image_loaded:
                self._btn_save.setEnabled(True)
                self._btn_spectrum.setEnabled(True)
                self._btn_reset.setEnabled(True)
                self._btn_info.setEnabled(True)
            self.statusBar().showMessage("就绪")

    def _run_async(self, func):
        if self._worker and self._worker.isRunning():
            return
        self._set_processing_state(True)
        self._worker = WorkerThread(func)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_result(self, payload):
        result, spectrum = payload
        self.show_image(result, self.label_dst)
        self.show_image(spectrum, self.label_spectrum)
        self._current_spectrum = spectrum
        self._fade_anim.stop()
        self._fade_anim.start()
        self._set_processing_state(False)

    def _on_error(self, msg: str):
        self._set_processing_state(False)
        QMessageBox.warning(self, "处理失败", msg)

    def show_image(self, img, label):
        if img is None:
            return
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qt_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._image_loaded:
            svc = self.controller.service
            if svc.image is not None:
                self.show_image(svc.image, self.label_src)
            if svc.result is not None:
                self.show_image(svc.result, self.label_dst)
            if self._current_spectrum is not None:
                self.show_image(self._current_spectrum, self.label_spectrum)

    # ── 槽函数 ───────────────────────────────────────────────────────────────

    def open_image(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "打开图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"
        )
        if not file:
            return
        payload = self.controller.open_image(file)
        if payload is None:
            QMessageBox.warning(self, "错误", "无法读取该图像文件")
            return
        result, spectrum = payload
        self._image_loaded = True
        self._current_spectrum = spectrum
        for c in self._need_image_cards:
            c.setEnabled(True)
        self._btn_save.setEnabled(True)
        self._btn_spectrum.setEnabled(True)
        self._btn_reset.setEnabled(True)
        self._btn_info.setEnabled(True)
        self.show_image(self.controller.service.image, self.label_src)
        self.show_image(result, self.label_dst)
        self.show_image(spectrum, self.label_spectrum)
        self.statusBar().showMessage(f"已打开：{os.path.basename(file)}")

    def save_result(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存结果", "result.png", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            result = self.controller.service.result
            if result is not None:
                ext = os.path.splitext(path)[1] or ".png"
                cv2.imencode(ext, result)[1].tofile(path)
                self.statusBar().showMessage(f"已保存结果：{os.path.basename(path)}")

    def save_spectrum(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存频谱", "spectrum.png", "PNG (*.png);;JPEG (*.jpg)")
        if path and self._current_spectrum is not None:
            ext = os.path.splitext(path)[1] or ".png"
            cv2.imencode(ext, self._current_spectrum)[1].tofile(path)
            self.statusBar().showMessage(f"已保存频谱：{os.path.basename(path)}")

    def reset(self):
        payload = self.controller.reset()
        if payload is None:
            return
        result, spectrum = payload
        self._current_spectrum = spectrum
        self.show_image(result, self.label_dst)
        self.show_image(spectrum, self.label_spectrum)
        self._fade_anim.stop()
        self._fade_anim.start()
        self.statusBar().showMessage("已恢复原图")

    def show_info(self):
        img = self.controller.service.image
        if img is not None:
            h, w = img.shape[:2]
            c = img.shape[2] if len(img.shape) == 3 else 1
            QMessageBox.information(self, "图像信息", f"宽度：{w} px\n高度：{h} px\n通道数：{c}")

    # ── 空间变换槽 ────────────────────────────────────────────────────────────

    def zoom_in(self):
        self._run_async(lambda: self.controller.zoom(2.0))

    def zoom_out(self):
        self._run_async(lambda: self.controller.zoom(0.5))

    def translate(self):
        tx, ok1 = QInputDialog.getInt(self, "平移", "X 方向偏移（像素）：", 50, -2000, 2000)
        if not ok1:
            return
        ty, ok2 = QInputDialog.getInt(self, "平移", "Y 方向偏移（像素）：", 30, -2000, 2000)
        if ok2:
            self._run_async(lambda: self.controller.translate(tx, ty))

    def rotate(self):
        angle, ok = QInputDialog.getDouble(self, "旋转", "旋转角度（度，逆时针为正）：", 30, -360, 360, 1)
        if ok:
            self._run_async(lambda: self.controller.rotate(angle))

    def gaussian_smooth(self):
        ksize, ok = QInputDialog.getInt(self, "高斯平滑", "核大小（奇数，3~51）：", 15, 3, 51, 2)
        if ok:
            self._run_async(lambda: self.controller.gaussian_smooth(ksize, 0))

    def laplacian_sharpen(self):
        self._run_async(self.controller.laplacian_sharpen)

    def median_denoise(self):
        ksize, ok = QInputDialog.getInt(self, "中值去噪", "核大小（奇数，3~21）：", 5, 3, 21, 2)
        if ok:
            self._run_async(lambda: self.controller.median_denoise(ksize))

    # ── 频率域滤波槽 ──────────────────────────────────────────────────────────

    def lowpass_filter(self):
        cutoff, ok = QInputDialog.getInt(self, "低通滤波", "截止频率（像素，5~200）：", 30, 5, 200)
        if ok:
            self._run_async(lambda: self.controller.butterworth_lowpass(cutoff, 2))

    def highpass_filter(self):
        cutoff, ok = QInputDialog.getInt(self, "高通滤波", "截止频率（像素，5~200）：", 30, 5, 200)
        if ok:
            self._run_async(lambda: self.controller.butterworth_highpass(cutoff, 2))
