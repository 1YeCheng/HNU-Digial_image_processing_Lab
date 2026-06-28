# -*- coding: utf-8 -*-
import os
import base64
import cv2
import numpy as np
from openai import OpenAI
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout,
    QScrollArea, QPushButton, QFileDialog, QMessageBox,
    QInputDialog, QSizePolicy, QStatusBar, QFrame,
    QTextEdit
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QCursor
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QPropertyAnimation,
    QEasingCurve, QSize
)
from PyQt5.QtWidgets import QGraphicsOpacityEffect

from controller.image_controller import ImageController

# ──────────────────────────────────────────────────────────────────────────── #
# Global QSS — Liveblocks-inspired premium dark theme
# ──────────────────────────────────────────────────────────────────────────── #
APP_STYLE = """
/* ── Root ── */
QMainWindow, QWidget {
    background-color: #FFFFFF;
    color: #333333;
    font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
    font-size: 13px;
}

/* ── Header bar ── */
QWidget#header {
    background-color: #FFFFFF;
    border-bottom: 1px solid #EEEEEE;
}

/* ── Image panels ── */
QLabel#img_label {
    background-color: #FAFAFA;
    border: 1px solid #EEEEEE;
    border-radius: 20px;
    color: #BBBBBB;
    font-size: 14px;
}

/* ── Panel title labels ── */
QLabel#panel_title {
    color: #AAAAAA;
    font-size: 11px;
    letter-spacing: 1px;
    background: transparent;
}

/* ── Section title in card strip ── */
QLabel#section_title {
    color: #AAAAAA;
    font-size: 10px;
    letter-spacing: 2px;
    background: transparent;
    padding-left: 4px;
}

/* ── Scroll areas (card strips) ── */
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:horizontal {
    background: transparent;
    height: 4px;
    border-radius: 2px;
}
QScrollBar::handle:horizontal {
    background: rgba(124,58,237,0.25);
    border-radius: 2px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: rgba(124,58,237,0.5);
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ── Primary action button (Open Image) ── */
QPushButton#btn_open {
    background-color: #7C3AED;
    color: #FFFFFF;
    border: none;
    border-radius: 20px;
    padding: 6px 20px;
    font-weight: 600;
    font-size: 13px;
}
QPushButton#btn_open:hover {
    background-color: #6D28D9;
}
QPushButton#btn_open:pressed {
    background-color: #5B21B6;
}

/* ── Secondary header buttons ── */
QPushButton#btn_save, QPushButton#btn_reset, QPushButton#btn_info {
    background-color: #F5F5F5;
    color: #555555;
    border: 1px solid #E8E8E8;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 13px;
}
QPushButton#btn_save:hover, QPushButton#btn_reset:hover, QPushButton#btn_info:hover {
    background-color: #EBEBEB;
    color: #333333;
    border-color: #DDDDDD;
}
QPushButton#btn_save:disabled, QPushButton#btn_reset:disabled, QPushButton#btn_info:disabled {
    color: #CCCCCC;
    border-color: #F0F0F0;
    background-color: #FAFAFA;
}

/* ── Status bar ── */
QStatusBar {
    background-color: #FFFFFF;
    color: #AAAAAA;
    font-size: 11px;
    border-top: 1px solid #EEEEEE;
}

/* ── AI review bar ── */
QWidget#ai_bar {
    background-color: #FAFAFA;
    border-bottom: 1px solid #EEEEEE;
}
QPushButton#btn_analyze {
    background-color: #EDE9FE;
    color: #7C3AED;
    border: none;
    border-radius: 10px;
    padding: 5px 16px;
    font-weight: 600;
    font-size: 12px;
}
QPushButton#btn_analyze:hover {
    background-color: #DDD6FE;
}
QPushButton#btn_analyze:disabled {
    background-color: #F5F5F5;
    color: #BBBBBB;
}
QTextEdit#ai_output {
    background-color: #FFFFFF;
    border: 1px solid #EEEEEE;
    border-radius: 10px;
    padding: 6px 10px;
    color: #444444;
    font-size: 12px;
}
"""

# Card QSS uses .format() placeholders — filled per group in _build_card_strip
CARD_NORMAL = """
    QWidget {{
        background-color: {bg};
        border: 1px solid {border};
        border-radius: 20px;
    }}
    QLabel {{ background: transparent; border: none; }}
"""
CARD_HOVER = """
    QWidget {{
        background-color: {hover_bg};
        border: 1px solid {hover_border};
        border-radius: 20px;
    }}
    QLabel {{ background: transparent; border: none; }}
"""
CARD_DISABLED = """
    QWidget {{
        background-color: #F8F8F8;
        border: 1px solid #F0F0F0;
        border-radius: 20px;
    }}
    QLabel {{ background: transparent; border: none; color: #CCCCCC; }}
"""

# Per-group colour tokens
GROUP_TOKENS = {
    "基础变换": {
        "strip_bg":    "#EBF5FF",
        "card_bg":     "#DBEAFE",
        "card_border": "#BFDBFE",
        "hover_bg":    "#BFDBFE",
        "hover_border":"#93C5FD",
    },
    "图像特效": {
        "strip_bg":    "#FFF0F6",
        "card_bg":     "#FCE7F3",
        "card_border": "#FBCFE8",
        "hover_bg":    "#FBCFE8",
        "hover_border":"#F9A8D4",
    },
    "空间变换": {
        "strip_bg":    "#F0FFF4",
        "card_bg":     "#DCFCE7",
        "card_border": "#BBF7D0",
        "hover_bg":    "#BBF7D0",
        "hover_border":"#86EFAC",
    },
}


# ──────────────────────────────────────────────────────────────────────────── #
# AnalyzeThread — calls GPT-5.4 vision API in background
# ──────────────────────────────────────────────────────────────────────────── #
_API_KEY = os.getenv("OPENAI_API_KEY")
_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

class AnalyzeThread(QThread):
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, img_bgr):
        super().__init__()
        self._img = img_bgr

    def run(self):
        try:
            if not _API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set; AI image review is disabled.")

            _, buf = cv2.imencode(".jpg", self._img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")

            client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL, timeout=150)
            response = client.chat.completions.create(
                model=_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "请用中文简洁地评价这张图像，涵盖：构图、色彩、清晰度、主体内容，"
                                "以及可以改进的地方。控制在100字以内。"
                            ),
                        },
                    ],
                }],
            )
            self.result_ready.emit(response.choices[0].message.content)
        except Exception as e:
            self.error_occurred.emit(str(e))


# ──────────────────────────────────────────────────────────────────────────── #
# WorkerThread — unchanged from original
# ──────────────────────────────────────────────────────────────────────────── #
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


# ──────────────────────────────────────────────────────────────────────────── #
# FeatureCard — clickable card widget with hover glow
# ──────────────────────────────────────────────────────────────────────────── #
class FeatureCard(QWidget):
    clicked = pyqtSignal()

    def __init__(self, icon: str, label: str, tokens: dict, parent=None):
        super().__init__(parent)
        self.setFixedSize(QSize(104, 80))
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self._enabled = True
        self._tokens = tokens  # group colour tokens for hover/normal states

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignCenter)

        self._icon_lbl = QLabel(icon)
        self._icon_lbl.setAlignment(Qt.AlignCenter)
        self._icon_lbl.setStyleSheet("font-size: 22px;")

        self._text_lbl = QLabel(label)
        self._text_lbl.setAlignment(Qt.AlignCenter)
        self._text_lbl.setWordWrap(True)
        self._text_lbl.setStyleSheet("font-size: 11px; color: #555555;")

        layout.addWidget(self._icon_lbl)
        layout.addWidget(self._text_lbl)

        self._apply_normal()

    def _apply_normal(self):
        t = self._tokens
        self.setStyleSheet(CARD_NORMAL.format(
            bg=t["card_bg"], border=t["card_border"]
        ))

    def _apply_hover(self):
        t = self._tokens
        self.setStyleSheet(CARD_HOVER.format(
            hover_bg=t["hover_bg"], hover_border=t["hover_border"]
        ))

    # ── hover ──
    def enterEvent(self, event):
        if self._enabled:
            self._apply_hover()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._enabled:
            self._apply_normal()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if self._enabled and event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    # ── enable / disable ──
    def setEnabled(self, enabled: bool):
        self._enabled = enabled
        self.setCursor(QCursor(Qt.PointingHandCursor if enabled else Qt.ForbiddenCursor))
        if enabled:
            self._apply_normal()
        else:
            self.setStyleSheet(CARD_DISABLED)


# ──────────────────────────────────────────────────────────────────────────── #
# MainWindow
# ──────────────────────────────────────────────────────────────────────────── #
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理教学系统")
        self.resize(1400, 820)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(APP_STYLE)

        self.controller = ImageController()
        self._worker = None
        self._analyze_worker = None
        self._image_loaded = False
        self._all_cards: list[FeatureCard] = []
        self._need_image_cards: list[FeatureCard] = []

        self.initUI()

    # ──────────────────────────────────────────────────────────────────────── #
    # UI construction
    # ──────────────────────────────────────────────────────────────────────── #
    def initUI(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        root.addWidget(self._build_ai_bar())
        root.addLayout(self._build_image_stage(), stretch=1)
        root.addWidget(self._build_card_strip())

        self.statusBar().showMessage("就绪  ·  请先打开一张图像")

    def _build_header(self) -> QWidget:
        """Top bar: title + action buttons."""
        header = QWidget()
        header.setObjectName("header")
        header.setFixedHeight(52)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(10)

        title = QLabel("图像处理教学系统")
        title.setStyleSheet(
            "font-size: 16px; font-weight: 700; "
            "color: #333333; background: transparent;"
        )

        # Divider
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
        self._btn_save.clicked.connect(self.save_image)

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
        layout.addWidget(self._btn_reset)
        layout.addWidget(self._btn_info)
        layout.addStretch()

        return header

    def _build_ai_bar(self) -> QWidget:
        """AI review bar: analyze button + output text."""
        bar = QWidget()
        bar.setObjectName("ai_bar")
        bar.setFixedHeight(56)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 8, 20, 8)
        layout.setSpacing(10)

        lbl = QLabel("✦ AI 图像评价")
        lbl.setStyleSheet(
            "font-size: 12px; font-weight: 600; color: #7C3AED; background: transparent;"
        )
        lbl.setFixedWidth(90)

        self._btn_analyze = QPushButton("分析图像")
        self._btn_analyze.setObjectName("btn_analyze")
        self._btn_analyze.setFixedHeight(32)
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.clicked.connect(self.analyze_image)

        self._ai_output = QTextEdit()
        self._ai_output.setObjectName("ai_output")
        self._ai_output.setReadOnly(True)
        self._ai_output.setPlaceholderText("打开图像后点击「分析图像」，AI 将对原始图像给出评价…")
        self._ai_output.setFixedHeight(40)
        self._ai_output.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        layout.addWidget(lbl)
        layout.addWidget(self._btn_analyze)
        layout.addWidget(self._ai_output, stretch=1)

        return bar

    def analyze_image(self):
        img = self.controller.service.image
        if img is None:
            return
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.setText("分析中…")
        self._ai_output.setPlainText("")

        self._analyze_worker = AnalyzeThread(img)
        self._analyze_worker.result_ready.connect(self._on_analyze_result)
        self._analyze_worker.error_occurred.connect(self._on_analyze_error)
        self._analyze_worker.start()

    def _on_analyze_result(self, text: str):
        self._ai_output.setPlainText(text)
        self._btn_analyze.setText("分析图像")
        self._btn_analyze.setEnabled(True)

    def _on_analyze_error(self, msg: str):
        self._ai_output.setPlainText(f"⚠ 请求失败：{msg}")
        self._btn_analyze.setText("分析图像")
        self._btn_analyze.setEnabled(True)

    def _build_image_stage(self) -> QHBoxLayout:
        """Side-by-side image panels."""
        layout = QHBoxLayout()
        layout.setContentsMargins(16, 12, 16, 8)
        layout.setSpacing(12)

        left = self._make_image_panel("原始图像")
        self.label_src = left.findChild(QLabel, "img_label")

        right = self._make_image_panel("处理结果")
        self.label_dst = right.findChild(QLabel, "img_label")

        # Fade-in animation on label_dst
        self._opacity_effect = QGraphicsOpacityEffect(self.label_dst)
        self._opacity_effect.setOpacity(1.0)
        self.label_dst.setGraphicsEffect(self._opacity_effect)

        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_anim.setDuration(280)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.OutCubic)

        layout.addWidget(left, 1)
        layout.addWidget(right, 1)
        return layout

    def _make_image_panel(self, title_text: str) -> QWidget:
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
        """Bottom panel: three horizontally-scrollable card rows."""
        container = QWidget()
        container.setFixedHeight(310)
        container.setStyleSheet("background-color: #F9FAFB; border-top: 1px solid #EEEEEE;")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(6)

        # ── Group definitions ──────────────────────────────────────────────
        groups = [
            ("基础变换", [
                ("🔲", "灰度化",    self.gray,          True),
                ("⬛", "二值化",    self.binary,         True),
                ("🔄", "反转",      self.inverse,        True),
                ("☀️", "Gamma",     self.gamma,          True),
                ("📈", "对数变换",  self.log_transform,  True),
                ("📉", "指数变换",  self.exp_transform,  True),
                ("📐", "缩小½",     self.resize_image,   True),
                ("📊", "均衡化",    self.hist_equalize,  True),
            ]),
            ("图像特效", [
                ("🌫️", "毛玻璃",   self.glass,          True),
                ("🗿",  "浮雕",     self.relief,         True),
                ("🎨", "油画",      self.oil,            True),
                ("🟦", "马赛克",    self.mask,           True),
                ("✏️", "素描",      self.sketch,         True),
                ("📷", "怀旧",      self.old,            True),
                ("💡", "光照",      self.lighting,       True),
                ("🎭", "卡通",      self.cartoonize,     True),
            ]),
            ("空间变换", [
                ("🔃", "旋转",      self.rotate,         True),
                ("↔️", "平移",      self.translate,      True),
                ("↩️", "水平镜像",  self.flip_h,         True),
                ("↕️", "垂直镜像",  self.flip_v,         True),
                ("🔍", "物理缩放",  self.zoom,           True),
                ("📐", "剪切",      self.shear,          True),
                ("🔭", "透视",      self.perspective,    True),
                ("〰️", "波浪",      self.wave,           True),
                ("🔗", "图像拼接",  self.stitch,         True),
            ]),
        ]

        for group_name, cards_def in groups:
            tokens = GROUP_TOKENS[group_name]

            row_widget = QWidget()
            row_widget.setStyleSheet(
                f"background-color: {tokens['strip_bg']}; "
                "border-radius: 16px;"
            )
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(10, 4, 10, 4)
            row_layout.setSpacing(8)

            # Section label
            sec_lbl = QLabel(group_name.upper())
            sec_lbl.setObjectName("section_title")
            sec_lbl.setFixedWidth(64)
            sec_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            row_layout.addWidget(sec_lbl)

            # Scrollable card area
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
                    card.setEnabled(False)  # disabled until image loaded

            inner_layout.addStretch()
            scroll.setWidget(inner)
            row_layout.addWidget(scroll)

            layout.addWidget(row_widget)

        return container

    # ──────────────────────────────────────────────────────────────────────── #
    # Utility helpers
    # ──────────────────────────────────────────────────────────────────────── #
    def _set_cards_enabled(self, enabled: bool):
        for card in self._all_cards:
            card.setEnabled(enabled)

    def _set_processing_state(self, processing: bool):
        if processing:
            self._set_cards_enabled(False)
            self._btn_save.setEnabled(False)
            self._btn_reset.setEnabled(False)
            self._btn_info.setEnabled(False)
            self.statusBar().showMessage("处理中，请稍候…")
        else:
            for card in self._all_cards:
                if card in self._need_image_cards:
                    card.setEnabled(self._image_loaded)
                else:
                    card.setEnabled(True)
            if self._image_loaded:
                self._btn_save.setEnabled(True)
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

    def _on_result(self, img):
        self.show_image(img, self.label_dst)
        self._fade_anim.stop()
        self._fade_anim.start()
        self._set_processing_state(False)

    def _on_error(self, msg):
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
            src = self.controller.service.image
            dst = self.controller.service.result
            if src is not None:
                self.show_image(src, self.label_src)
            if dst is not None:
                self.show_image(dst, self.label_dst)

    # ──────────────────────────────────────────────────────────────────────── #
    # Slot functions — logic identical to original
    # ──────────────────────────────────────────────────────────────────────── #
    def open_image(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "打开图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"
        )
        if file:
            img = self.controller.open_image(file)
            if img is None:
                QMessageBox.warning(self, "错误", "无法读取该图像文件")
                return
            self._image_loaded = True
            for card in self._need_image_cards:
                card.setEnabled(True)
            self._btn_save.setEnabled(True)
            self._btn_reset.setEnabled(True)
            self._btn_info.setEnabled(True)
            self._btn_analyze.setEnabled(True)
            self.show_image(img, self.label_src)
            self.show_image(img, self.label_dst)
            self.statusBar().showMessage(f"已打开：{os.path.basename(file)}")

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "result.png", "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            # get current result from service via controller
            info = self.controller.get_info()
            if info is None:
                return
            # retrieve result image through a reset-safe approach
            result = self.controller.service.result
            if result is not None:
                cv2.imencode(os.path.splitext(path)[1] or '.png', result)[1].tofile(path)
                self.statusBar().showMessage(f"已保存：{os.path.basename(path)}")

    def show_info(self):
        info = self.controller.get_info()
        if info:
            w, h, c = info
            QMessageBox.information(self, "图像信息", f"宽度：{w} px\n高度：{h} px\n通道数：{c}")
        else:
            QMessageBox.warning(self, "提示", "请先打开图像")

    def reset(self):
        img = self.controller.reset()
        if img is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        self.show_image(img, self.label_dst)
        self._fade_anim.stop()
        self._fade_anim.start()
        self.statusBar().showMessage("已恢复原图")

    def resize_image(self):
        self._run_async(self.controller.resize_half)

    def gray(self):
        self._run_async(self.controller.gray)

    def binary(self):
        self._run_async(self.controller.binary)

    def inverse(self):
        self._run_async(self.controller.inverse)

    def gamma(self):
        self._run_async(self.controller.gamma)

    def log_transform(self):
        self._run_async(self.controller.log_transform)

    def exp_transform(self):
        self._run_async(self.controller.exp_transform)

    def hist_equalize(self):
        self._run_async(self.controller.hist_equalize)

    def glass(self):
        self._run_async(self.controller.glass)

    def relief(self):
        self._run_async(self.controller.relief)

    def oil(self):
        self._run_async(self.controller.oil)

    def mask(self):
        self._run_async(self.controller.mask)

    def sketch(self):
        self._run_async(self.controller.sketch)

    def old(self):
        self._run_async(self.controller.old)

    def lighting(self):
        self._run_async(self.controller.lighting)

    def cartoonize(self):
        self._run_async(self.controller.cartoonize)

    def rotate(self):
        angle, ok = QInputDialog.getInt(self, "旋转", "旋转角度（度）：", 30, -360, 360)
        if ok:
            self._run_async(lambda: self.controller.rotate(angle))

    def translate(self):
        tx, ok1 = QInputDialog.getInt(self, "平移", "X 方向偏移（像素）：", 50, -2000, 2000)
        if not ok1:
            return
        ty, ok2 = QInputDialog.getInt(self, "平移", "Y 方向偏移（像素）：", 30, -2000, 2000)
        if ok2:
            self._run_async(lambda: self.controller.translate(tx, ty))

    def flip_h(self):
        self._run_async(self.controller.flip_h)

    def flip_v(self):
        self._run_async(self.controller.flip_v)

    def zoom(self):
        factor, ok = QInputDialog.getDouble(
            self, "物理缩放", "缩放因子（<1 缩小，>1 放大）：", 0.5, 0.01, 10.0, 2
        )
        if ok:
            self._run_async(lambda: self.controller.zoom(factor))

    def shear(self):
        factor, ok = QInputDialog.getDouble(
            self, "剪切变换", "剪切因子：", 0.3, -5.0, 5.0, 2
        )
        if ok:
            self._run_async(lambda: self.controller.shear(factor))

    def perspective(self):
        self._run_async(self.controller.perspective)

    def wave(self):
        self._run_async(self.controller.wave)

    def stitch(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "选择要拼接的图片", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if not file:
            return
        img2 = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img2 is None:
            QMessageBox.warning(self, "错误", "无法读取该图像文件")
            return

        def _do_stitch():
            result, ok = self.controller.stitch(img2)
            return result

        self._set_processing_state(True)
        self._worker = WorkerThread(_do_stitch)
        self._worker.result_ready.connect(self._on_stitch_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_stitch_result(self, img):
        self.show_image(img, self.label_dst)
        self._fade_anim.stop()
        self._fade_anim.start()
        self._set_processing_state(False)
