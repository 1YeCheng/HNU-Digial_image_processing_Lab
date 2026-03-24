import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from controller.image_controller import ImageController


class WorkerThread(QThread):
    """后台处理线程，避免耗时操作阻塞 UI"""
    result_ready = pyqtSignal(object)  # 处理完成后发送结果图像
    error_occurred = pyqtSignal(str)   # 处理出错时发送错误信息

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
        self.setWindowTitle("图像处理教学系统(多层架构)")
        self.resize(1200, 700)
        self.controller = ImageController()
        self._worker = None
        self.initUI()

    def initUI(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout()

        self.label_src = QLabel("原始图像")
        self.label_src.setAlignment(Qt.AlignCenter)
        self.label_src.setStyleSheet("border:1px solid gray")
        self.label_src.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.label_dst = QLabel("处理结果")
        self.label_dst.setAlignment(Qt.AlignCenter)
        self.label_dst.setStyleSheet("border:1px solid gray")
        self.label_dst.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        control = self.create_buttons()

        layout.addWidget(self.label_src)
        layout.addWidget(self.label_dst)
        layout.addWidget(control)

        layout.setStretch(0, 4)
        layout.setStretch(1, 4)
        layout.setStretch(2, 1)

        central.setLayout(layout)

        # 状态栏
        self.statusBar().showMessage("就绪")

    def create_buttons(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(160)

        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)

        self._buttons_def = [
            ("打开图像",   self.open_image,  False),
            ("图像信息",   self.show_info,   False),
            ("恢复原图",   self.reset,       False),
            ("缩小一半",   self.resize_image, True),
            ("灰度化",     self.gray,         True),
            ("二值化",     self.binary,       True),
            ("反转",       self.inverse,      True),
            ("Gamma变换",  self.gamma,        True),
            ("对数变换",   self.log_transform, True),
            ("指数变换",   self.exp_transform, True),
            ("毛玻璃特效", self.glass,        True),
            ("浮雕特效",   self.relief,       True),
            ("油画特效",   self.oil,          True),
            ("马赛克特效", self.mask,         True),
            ("素描特效",   self.sketch,       True),
            ("怀旧特效",   self.old,          True),
            ("光照特效",   self.lighting,     True),
            ("卡通特效",   self.cartoonize,   True),
        ]

        self._all_btns = []
        for text, func, _ in self._buttons_def:
            btn = QPushButton(text)
            btn.setMinimumHeight(38)
            btn.clicked.connect(func)
            layout.addWidget(btn)
            self._all_btns.append(btn)

        layout.addStretch()
        panel.setLayout(layout)
        scroll.setWidget(panel)
        return scroll

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #

    def _set_buttons_enabled(self, enabled: bool):
        for btn in self._all_btns:
            btn.setEnabled(enabled)

    def _run_async(self, func):
        """在后台线程执行 func()，完成后更新右侧图像"""
        if self._worker and self._worker.isRunning():
            return
        self._set_buttons_enabled(False)
        self.statusBar().showMessage("处理中，请稍候...")
        self._worker = WorkerThread(func)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_result(self, img):
        self.show_image(img, self.label_dst)
        self._set_buttons_enabled(True)
        self.statusBar().showMessage("就绪")

    def _on_error(self, msg):
        self._set_buttons_enabled(True)
        self.statusBar().showMessage("就绪")
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
            self.show_image(img, self.label_src)
            self.show_image(img, self.label_dst)
            self.statusBar().showMessage(f"已打开：{file}")

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
        self.show_image(img, self.label_dst)

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
