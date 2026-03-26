import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from controller.image_controller import ImageController

# ------------------------------------------------------------------ #
# 全局样式表
# ------------------------------------------------------------------ #
APP_STYLE = """
QMainWindow {
    background-color: #fffbe6;
}
QWidget {
    background-color: #fffbe6;
    color: #1a2340;
    font-family: 'Microsoft YaHei', Arial, sans-serif;
    font-size: 13px;
}
QLabel#img_label {
    background-color: #fff8d0;
    border: 2px solid #e6c84a;
    border-radius: 8px;
    color: #b8a000;
    font-size: 14px;
}
QScrollArea {
    border: none;
    background-color: #fff4b0;
}
QScrollBar:vertical {
    background: #fff4b0;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #4a78c8;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #2255aa;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QPushButton {
    background-color: #3a6fd8;
    color: #ffffff;
    border: 1px solid #2255aa;
    border-radius: 6px;
    padding: 6px 4px;
    text-align: center;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2255aa;
    border: 1px solid #1a3a80;
    color: #ffe066;
}
QPushButton:pressed {
    background-color: #1a3a80;
    color: #ffe066;
}
QPushButton:disabled {
    background-color: #b8c8e8;
    color: #8899bb;
    border: 1px solid #c8d8f0;
}
QPushButton#btn_open {
    background-color: #1a3a80;
    color: #ffe066;
    font-weight: bold;
    border: none;
}
QPushButton#btn_open:hover {
    background-color: #0d2255;
    color: #ffffff;
}
QPushButton#btn_reset {
    background-color: #e05a2b;
    color: #ffffff;
    font-weight: bold;
    border: none;
}
QPushButton#btn_reset:hover {
    background-color: #b84020;
}
QStatusBar {
    background-color: #fff4b0;
    color: #5a6a40;
    font-size: 12px;
    border-top: 1px solid #e6c84a;
}
"""


class WorkerThread(QThread):
    """后台处理线程，避免耗时操作阻塞 UI"""
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, func):
        super().__init__()
        self._func = func

    def run(self):
        try:
            result = self._func()
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理教学系统")
        self.resize(1280, 760)
        self.setStyleSheet(APP_STYLE)
        self.controller = ImageController()
        self._worker = None
        self._image_loaded = False
        self.initUI()

    def initUI(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 顶部标题栏
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet("background-color: #181825; border-bottom: 1px solid #313244;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 0, 16, 0)
        icon_label = QLabel("🖼")
        icon_label.setStyleSheet("font-size: 20px; background: transparent;")
        title = QLabel("图像处理教学系统")
        title.setObjectName("title_label")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa; background: transparent;")
        header_layout.addWidget(icon_label)
        header_layout.addWidget(title)
        header_layout.addStretch()
        root.addWidget(header)

        # 主体内容区
        body = QHBoxLayout()
        body.setContentsMargins(12, 12, 12, 12)
        body.setSpacing(12)

        # 左侧：原始图像
        left_panel = self._make_image_panel("原始图像")
        self.label_src = left_panel.findChild(QLabel, "img_label")

        # 中间：处理结果
        right_panel = self._make_image_panel("处理结果")
        self.label_dst = right_panel.findChild(QLabel, "img_label")

        # 右侧：按钮面板
        control = self._create_buttons()

        body.addWidget(left_panel, 4)
        body.addWidget(right_panel, 4)
        body.addWidget(control, 0)

        root.addLayout(body)
        central.setLayout(root)

        self.statusBar().showMessage("就绪  |  请先打开一张图像")

    def _make_image_panel(self, title_text):
        """创建带标题的图像显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel(title_text)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "color: #a6adc8; font-size: 13px; "
            "background-color: #181825; "
            "border-radius: 6px 6px 0 0; padding: 4px;"
        )

        img_label = QLabel(title_text)
        img_label.setObjectName("img_label")
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(title)
        layout.addWidget(img_label)
        return panel

    def _create_buttons(self):
        """创建右侧按钮面板"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(148)

        panel = QWidget()
        panel.setStyleSheet("background-color: #181825;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(5)

        # (显示文字, 槽函数, 需要图像才能用)
        self._buttons_def = [
            ("打开图像",     self.open_image,     False, "btn_open"),
            ("图像信息",     self.show_info,      True,  ""),
            ("恢复原图",     self.reset,          True,  "btn_reset"),
            (None, None, None, None),  # 分隔
            ("缩小一半",     self.resize_image,   True,  ""),
            ("灰度化",       self.gray,           True,  ""),
            ("二值化",       self.binary,         True,  ""),
            ("反转",         self.inverse,        True,  ""),
            ("Gamma变换",    self.gamma,          True,  ""),
            ("对数变换",     self.log_transform,  True,  ""),
            ("指数变换",     self.exp_transform,  True,  ""),
            ("直方图均衡化", self.hist_equalize,  True,  ""),
            (None, None, None, None),  # 分隔
            ("毛玻璃特效",   self.glass,          True,  ""),
            ("浮雕特效",     self.relief,         True,  ""),
            ("油画特效",     self.oil,            True,  ""),
            ("马赛克特效",   self.mask,           True,  ""),
            ("素描特效",     self.sketch,         True,  ""),
            ("怀旧特效",     self.old,            True,  ""),
            ("光照特效",     self.lighting,       True,  ""),
            ("卡通特效",     self.cartoonize,     True,  ""),
        ]

        self._all_btns = []
        self._need_image_btns = []

        for item in self._buttons_def:
            text, func, need_img, obj_name = item
            if text is None:
                sep = QFrame()
                sep.setFrameShape(QFrame.HLine)
                sep.setStyleSheet("color: #313244; margin: 2px 0;")
                layout.addWidget(sep)
                continue
            btn = QPushButton(text)
            btn.setMinimumHeight(36)
            if obj_name:
                btn.setObjectName(obj_name)
            btn.clicked.connect(func)
            layout.addWidget(btn)
            self._all_btns.append(btn)
            if need_img:
                self._need_image_btns.append(btn)
                btn.setEnabled(False)  # 初始禁用，等图像加载后启用

        layout.addStretch()
        scroll.setWidget(panel)
        return scroll

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #

    def _set_buttons_enabled(self, enabled: bool):
        for btn in self._all_btns:
            btn.setEnabled(enabled)

    def _set_processing_state(self, processing: bool):
        """处理中：禁用所有按钮；处理完：恢复需要图像的按钮"""
        if processing:
            self._set_buttons_enabled(False)
            self.statusBar().showMessage("处理中，请稍候...")
        else:
            # 打开图像按钮始终可用
            for btn in self._all_btns:
                if btn.objectName() == "btn_open":
                    btn.setEnabled(True)
                elif btn in self._need_image_btns:
                    btn.setEnabled(self._image_loaded)
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

    # ------------------------------------------------------------------ #
    # 槽函数
    # ------------------------------------------------------------------ #

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
            for btn in self._need_image_btns:
                btn.setEnabled(True)
            self.show_image(img, self.label_src)
            self.show_image(img, self.label_dst)
            import os
            self.statusBar().showMessage(f"已打开：{os.path.basename(file)}")

    def show_info(self):
        info = self.controller.get_info()
        if info:
            w, h, c = info
            QMessageBox.information(
                self, "图像信息",
                f"宽度：{w} px\n高度：{h} px\n通道数：{c}"
            )
        else:
            QMessageBox.warning(self, "提示", "请先打开图像")

    def reset(self):
        img = self.controller.reset()
        if img is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        self.show_image(img, self.label_dst)
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
