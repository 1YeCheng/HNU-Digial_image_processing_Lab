import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QGroupBox, QGridLayout, QMessageBox, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class MorphologyProcessor:
    """
    数学形态学处理器类
    """

    def __init__(self):
        self.original_image = None  # 原始图像 (BGR)
        self.gray_image = None  # 灰度图像
        self.binary_image = None  # 二值图像 (0或255)
        self.kernel_size = 3  # 结构元素大小
        self.kernel_shape = cv2.MORPH_RECT  # 结构元素形状

    def set_kernel(self, size=3, shape=cv2.MORPH_RECT):
        self.kernel_size = size
        self.kernel_shape = shape
        return cv2.getStructuringElement(shape, (size, size))

    def get_kernel(self):
        return cv2.getStructuringElement(self.kernel_shape, (self.kernel_size, self.kernel_size))

    def load_image(self, filepath):
        """
        加载图像（已支持中文路径）
        """
        try:
            # 修改点：不再直接使用 cv2.imread(filepath)
            # 而是先用 numpy 读取原始字节流，再由 opencv 解码
            file_data = np.fromfile(filepath, dtype=np.uint8)
            self.original_image = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"读取失败: {e}")
            return False

        if self.original_image is None:
            return False

        # 转换为灰度图
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.binary_image = None
        return True

    def adaptive_threshold(self, block_size=11, C=2):
        if self.gray_image is None:
            return None
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3

        binary = cv2.adaptiveThreshold(self.gray_image, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       block_size, C)
        self.binary_image = binary
        return self.binary_image

    # --- 以下形态学运算方法保持不变 ---
    def dilate(self, iterations=1):
        if self.binary_image is None: return None
        return cv2.dilate(self.binary_image, self.get_kernel(), iterations=iterations)

    def erode(self, iterations=1):
        if self.binary_image is None: return None
        return cv2.erode(self.binary_image, self.get_kernel(), iterations=iterations)

    def open(self, iterations=1):
        if self.binary_image is None: return None
        return cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, self.get_kernel(), iterations=iterations)

    def close(self, iterations=1):
        if self.binary_image is None: return None
        return cv2.morphologyEx(self.binary_image, cv2.MORPH_CLOSE, self.get_kernel(), iterations=iterations)

    def boundary(self):
        if self.binary_image is None: return None
        kernel = self.get_kernel()
        dilated = cv2.dilate(self.binary_image, kernel)
        eroded = cv2.erode(self.binary_image, kernel)
        return cv2.subtract(dilated, eroded)

    def fill_holes(self):
        if self.binary_image is None: return None
        im_floodfill = self.binary_image.copy()
        h, w = self.binary_image.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        return self.binary_image | im_floodfill_inv

    def top_hat(self):
        if self.gray_image is None: return None
        return cv2.morphologyEx(self.gray_image, cv2.MORPH_TOPHAT, self.get_kernel())

    def black_hat(self):
        if self.gray_image is None: return None
        return cv2.morphologyEx(self.gray_image, cv2.MORPH_BLACKHAT, self.get_kernel())

    def hit_or_miss(self, custom_kernel=None):
        if self.binary_image is None: return None
        if custom_kernel is None:
            kernel = np.zeros((3, 3), dtype=np.int8)
            kernel[0, 0], kernel[0, 1], kernel[1, 0] = 1, 1, 1
            kernel[1, 1] = -1
        else:
            kernel = custom_kernel
        return cv2.morphologyEx(self.binary_image, cv2.MORPH_HITMISS, kernel)


class MorphologyUI(QMainWindow):
    """
    UI界面类保持逻辑基本不变，仅优化显示
    """

    def __init__(self):
        super().__init__()
        self.processor = MorphologyProcessor()
        self.current_result_image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("数学形态学图像处理系统 (支持中文路径)")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(280)

        file_group = QGroupBox("1. 文件操作")
        file_layout = QVBoxLayout()
        self.btn_open_file = QPushButton("选择图片 (支持中文)")
        self.btn_open_file.clicked.connect(self.open_image)
        file_layout.addWidget(self.btn_open_file)
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)

        binary_group = QGroupBox("2. 前置处理")
        binary_layout = QVBoxLayout()
        self.btn_adaptive_thresh = QPushButton("自适应二值化")
        self.btn_adaptive_thresh.clicked.connect(self.adaptive_threshold)
        binary_layout.addWidget(self.btn_adaptive_thresh)
        binary_group.setLayout(binary_layout)
        control_layout.addWidget(binary_group)

        morph_group = QGroupBox("3. 形态学运算")
        morph_grid = QGridLayout()
        self.btns = {
            "腐蚀": self.erode_operation,
            "膨胀": self.dilate_operation,
            "开运算": self.open_operation,
            "闭运算": self.close_operation,
            "求边界": self.boundary_operation,
            "孔洞填充": self.fill_operation,
            "高帽(亮细节)": self.tophat_operation,
            "黑帽(暗细节)": self.blackhat_operation,
            "击中击不中": self.hitmiss_operation
        }

        row, col = 0, 0
        for name, func in self.btns.items():
            btn = QPushButton(name)
            btn.clicked.connect(func)
            btn.setEnabled(False)
            setattr(self, f"btn_{name}", btn)  # 方便后续启用
            morph_grid.addWidget(btn, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1
        morph_group.setLayout(morph_grid)
        control_layout.addWidget(morph_group)
        control_layout.addStretch()

        # 显示面板
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)

        self.label_original = QLabel("请加载原始图像")
        self.label_result = QLabel("等待运算结果")
        for lbl in [self.label_original, self.label_result]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background: #333; color: #999; border: 2px dashed #555;")

        scroll_orig = QScrollArea()
        scroll_orig.setWidget(self.label_original)
        scroll_orig.setWidgetResizable(True)

        scroll_res = QScrollArea()
        scroll_res.setWidget(self.label_result)
        scroll_res.setWidgetResizable(True)

        display_layout.addWidget(QLabel("原始图像预览:"))
        display_layout.addWidget(scroll_orig)
        display_layout.addWidget(QLabel("运算结果预览:"))
        display_layout.addWidget(scroll_res)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(display_widget, 1)

    def set_buttons_enabled(self, enabled):
        self.btn_adaptive_thresh.setEnabled(enabled)
        for name in self.btns.keys():
            getattr(self, f"btn_{name}").setEnabled(enabled)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            if self.processor.load_image(file_path):
                self.display_image(self.processor.original_image, self.label_original, is_bgr=True)
                self.set_buttons_enabled(True)
                self.label_result.setText("图片加载成功，请先进行二值化或直接灰度形态学变换")
            else:
                QMessageBox.critical(self, "错误", "无法读取该文件，请检查路径或文件完整性。")

    def display_image(self, img, label, is_bgr=False):
        if img is None: return
        # 转换颜色空间
        if is_bgr:
            img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            img_show = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_show = img

        h, w, ch = img_show.shape
        bytes_per_line = ch * w
        qimg = QImage(img_show.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # 根据Label大小缩放展示，但不改变滚动区域内的实际尺寸感
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    # --- 槽函数映射 ---
    def adaptive_threshold(self):
        res = self.processor.adaptive_threshold()
        self.display_image(res, self.label_result)

    def erode_operation(self):
        self.display_image(self.processor.erode(), self.label_result)

    def dilate_operation(self):
        self.display_image(self.processor.dilate(), self.label_result)

    def open_operation(self):
        self.display_image(self.processor.open(), self.label_result)

    def close_operation(self):
        self.display_image(self.processor.close(), self.label_result)

    def boundary_operation(self):
        self.display_image(self.processor.boundary(), self.label_result)

    def fill_operation(self):
        self.display_image(self.processor.fill_holes(), self.label_result)

    def tophat_operation(self):
        self.display_image(self.processor.top_hat(), self.label_result)

    def blackhat_operation(self):
        self.display_image(self.processor.black_hat(), self.label_result)

    def hitmiss_operation(self):
        self.display_image(self.processor.hit_or_miss(), self.label_result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MorphologyUI()
    ex.show()
    sys.exit(app.exec_())