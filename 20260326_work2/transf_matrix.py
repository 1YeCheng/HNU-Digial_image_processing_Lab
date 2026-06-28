import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import math


class ImageTransformDemo(QMainWindow):
    """图像空间变换教学演示系统，支持缩放、旋转、剪切、透视、波浪特效等变换"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("图像空间变换教学演示系统 - 增强版")
        self.resize(1600, 900)  # 增大到1600x900

        # 设置窗口最小尺寸，防止缩得太小
        self.setMinimumSize(1200, 700)

        self.image = None  # 原始图像 (BGR)
        self.result = None  # 当前显示的图像 (BGR)
        self.original_size = None  # 记录原始尺寸用于还原

        # 记录当前变换信息
        self.current_transform = {
            'type': '原始图像',
            'matrix': None,
            'params': {}
        }

        # 自定义仿射变换点
        self.src_points = []
        self.dst_points = []
        self.affine_selection_mode = False

        self.initUI()

    def initUI(self):
        """初始化用户界面"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # 左侧：原始图像显示区域
        src_group = QGroupBox("原始图像")
        src_layout = QVBoxLayout()
        self.label_src = QLabel("未加载图像")
        self.label_src.setAlignment(Qt.AlignCenter)
        self.label_src.setStyleSheet("border: 2px solid #aaa; background-color: #f5f5f5;")
        self.label_src.setMinimumSize(450, 450)
        self.label_src.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        src_layout.addWidget(self.label_src)
        src_group.setLayout(src_layout)

        # 中间：变换结果图像显示区域
        dst_group = QGroupBox("变换结果")
        dst_layout = QVBoxLayout()
        self.label_dst = QLabel("等待变换")
        self.label_dst.setAlignment(Qt.AlignCenter)
        self.label_dst.setStyleSheet("border: 2px solid #aaa; background-color: #f5f5f5;")
        self.label_dst.setMinimumSize(450, 450)
        self.label_dst.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        dst_layout.addWidget(self.label_dst)
        dst_group.setLayout(dst_layout)

        # 右侧：控制面板（分页形式）
        control_panel = self.create_control_panel()
        control_panel.setMaximumWidth(350)
        control_panel.setMinimumWidth(320)

        main_layout.addWidget(src_group, 4)
        main_layout.addWidget(dst_group, 4)
        main_layout.addWidget(control_panel, 2)

        # 设置比例
        main_layout.setStretch(0, 4)
        main_layout.setStretch(1, 4)
        main_layout.setStretch(2, 2)

    def create_control_panel(self):
        """创建包含所有变换按钮的控制面板（使用选项卡）"""
        panel = QWidget()
        layout = QVBoxLayout()

        # 创建选项卡
        tabs = QTabWidget()

        # ========== 文件操作选项卡 ==========
        file_tab = self.create_file_tab()
        tabs.addTab(file_tab, "📁 文件操作")

        # ========== 基础变换选项卡 ==========
        basic_tab = self.create_basic_transform_tab()
        tabs.addTab(basic_tab, "🔄 基础变换")

        # ========== 高级变换选项卡 ==========
        advanced_tab = self.create_advanced_transform_tab()
        tabs.addTab(advanced_tab, "✨ 高级变换")

        # ========== 矩阵信息选项卡 ==========
        matrix_tab = self.create_matrix_info_tab()
        tabs.addTab(matrix_tab, "📊 变换矩阵")

        layout.addWidget(tabs)
        panel.setLayout(layout)
        return panel

    def create_file_tab(self):
        """创建文件操作选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        btn_open = QPushButton("📂 打开图像")
        btn_open.clicked.connect(self.open_image)
        btn_open.setMinimumHeight(45)

        btn_info = QPushButton("ℹ️ 图像信息")
        btn_info.clicked.connect(self.show_info)
        btn_info.setMinimumHeight(45)

        btn_reset = QPushButton("🔄 还原原始图像")
        btn_reset.clicked.connect(self.reset_image)
        btn_reset.setMinimumHeight(45)
        btn_reset.setStyleSheet("background-color: #e8f0fe; font-weight: bold;")

        btn_save = QPushButton("💾 保存结果图像")
        btn_save.clicked.connect(self.save_image)
        btn_save.setMinimumHeight(45)

        layout.addWidget(btn_open)
        layout.addWidget(btn_info)
        layout.addWidget(btn_reset)
        layout.addWidget(btn_save)
        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_basic_transform_tab(self):
        """创建基础变换选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # 缩放变换
        zoom_group = QGroupBox("缩放变换")
        zoom_layout = QVBoxLayout()
        self.zoom_factor = QDoubleSpinBox()
        self.zoom_factor.setRange(0.1, 3.0)
        self.zoom_factor.setSingleStep(0.1)
        self.zoom_factor.setValue(0.5)
        self.zoom_factor.setPrefix("缩放因子: ")
        self.zoom_factor.setSuffix(" 倍")
        btn_zoom = QPushButton("应用缩放")
        btn_zoom.clicked.connect(self.zoom_transform)
        zoom_layout.addWidget(self.zoom_factor)
        zoom_layout.addWidget(btn_zoom)
        zoom_group.setLayout(zoom_layout)

        # 旋转变换
        rotate_group = QGroupBox("旋转变换")
        rotate_layout = QVBoxLayout()
        self.rotate_angle = QSpinBox()
        self.rotate_angle.setRange(-360, 360)
        self.rotate_angle.setSingleStep(15)
        self.rotate_angle.setValue(30)
        self.rotate_angle.setPrefix("旋转角度: ")
        self.rotate_angle.setSuffix(" °")
        btn_rotate = QPushButton("应用旋转")
        btn_rotate.clicked.connect(self.rotate_transform)
        rotate_layout.addWidget(self.rotate_angle)
        rotate_layout.addWidget(btn_rotate)
        rotate_group.setLayout(rotate_layout)

        # 平移变换
        trans_group = QGroupBox("平移变换")
        trans_layout = QVBoxLayout()
        trans_input_layout = QHBoxLayout()
        self.tx = QSpinBox()
        self.tx.setRange(-500, 500)
        self.tx.setValue(50)
        self.tx.setPrefix("Δx: ")
        self.ty = QSpinBox()
        self.ty.setRange(-500, 500)
        self.ty.setValue(30)
        self.ty.setPrefix("Δy: ")
        trans_input_layout.addWidget(self.tx)
        trans_input_layout.addWidget(self.ty)
        btn_translate = QPushButton("应用平移")
        btn_translate.clicked.connect(self.translate_transform)
        trans_layout.addLayout(trans_input_layout)
        trans_layout.addWidget(btn_translate)
        trans_group.setLayout(trans_layout)

        # 镜像翻转
        flip_group = QGroupBox("镜像翻转")
        flip_layout = QHBoxLayout()
        btn_flip_h = QPushButton("水平镜像")
        btn_flip_h.clicked.connect(self.flip_horizontal)
        btn_flip_v = QPushButton("垂直镜像")
        btn_flip_v.clicked.connect(self.flip_vertical)
        flip_layout.addWidget(btn_flip_h)
        flip_layout.addWidget(btn_flip_v)
        flip_group.setLayout(flip_layout)

        layout.addWidget(zoom_group)
        layout.addWidget(rotate_group)
        layout.addWidget(trans_group)
        layout.addWidget(flip_group)
        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_advanced_transform_tab(self):
        """创建高级变换选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # 剪切变换
        shear_group = QGroupBox("剪切变换")
        shear_layout = QVBoxLayout()

        shear_h_layout = QHBoxLayout()
        self.shear_factor = QDoubleSpinBox()
        self.shear_factor.setRange(-1.5, 1.5)
        self.shear_factor.setSingleStep(0.1)
        self.shear_factor.setValue(0.3)
        self.shear_factor.setPrefix("水平: ")
        btn_shear = QPushButton("水平剪切")
        btn_shear.clicked.connect(self.shear_transform)
        shear_h_layout.addWidget(self.shear_factor)
        shear_h_layout.addWidget(btn_shear)

        shear_v_layout = QHBoxLayout()
        self.shear_v_factor = QDoubleSpinBox()
        self.shear_v_factor.setRange(-1.5, 1.5)
        self.shear_v_factor.setSingleStep(0.1)
        self.shear_v_factor.setValue(0.2)
        self.shear_v_factor.setPrefix("垂直: ")
        btn_shear_v = QPushButton("垂直剪切")
        btn_shear_v.clicked.connect(self.shear_v_transform)
        shear_v_layout.addWidget(self.shear_v_factor)
        shear_v_layout.addWidget(btn_shear_v)

        shear_layout.addLayout(shear_h_layout)
        shear_layout.addLayout(shear_v_layout)
        shear_group.setLayout(shear_layout)

        # 透视变换
        perspective_group = QGroupBox("透视变换")
        perspective_layout = QVBoxLayout()

        btn_perspective_preset = QPushButton("📐 预设透视（梯形）")
        btn_perspective_preset.clicked.connect(self.perspective_preset)

        btn_perspective_custom = QPushButton("🎯 自定义透视变换")
        btn_perspective_custom.clicked.connect(self.custom_perspective)

        perspective_layout.addWidget(btn_perspective_preset)
        perspective_layout.addWidget(btn_perspective_custom)
        perspective_group.setLayout(perspective_layout)

        # 波浪扭曲特效
        wave_group = QGroupBox("波浪扭曲特效")
        wave_layout = QVBoxLayout()

        wave_param_layout = QGridLayout()
        self.wave_amplitude = QDoubleSpinBox()
        self.wave_amplitude.setRange(1, 50)
        self.wave_amplitude.setValue(15)
        self.wave_amplitude.setPrefix("振幅: ")

        self.wave_frequency = QDoubleSpinBox()
        self.wave_frequency.setRange(0.01, 0.5)
        self.wave_frequency.setSingleStep(0.01)
        self.wave_frequency.setValue(0.1)
        self.wave_frequency.setPrefix("频率: ")

        wave_param_layout.addWidget(QLabel("振幅:"), 0, 0)
        wave_param_layout.addWidget(self.wave_amplitude, 0, 1)
        wave_param_layout.addWidget(QLabel("频率:"), 1, 0)
        wave_param_layout.addWidget(self.wave_frequency, 1, 1)

        btn_wave_h = QPushButton("🌊 水平波浪")
        btn_wave_h.clicked.connect(self.wave_transform_horizontal)
        btn_wave_v = QPushButton("🌊 垂直波浪")
        btn_wave_v.clicked.connect(self.wave_transform_vertical)

        wave_layout.addLayout(wave_param_layout)
        wave_layout.addWidget(btn_wave_h)
        wave_layout.addWidget(btn_wave_v)
        wave_group.setLayout(wave_layout)

        # 自定义仿射变换
        affine_group = QGroupBox("自定义仿射变换")
        affine_layout = QVBoxLayout()

        btn_affine_start = QPushButton("🎨 开始自定义仿射变换")
        btn_affine_start.clicked.connect(self.start_affine_selection)
        btn_affine_start.setStyleSheet("background-color: #fff3e0;")

        self.affine_status = QLabel("状态：未开始")
        self.affine_status.setWordWrap(True)
        self.affine_status.setStyleSheet("color: gray; font-size: 10px;")

        affine_layout.addWidget(btn_affine_start)
        affine_layout.addWidget(self.affine_status)
        affine_group.setLayout(affine_layout)

        layout.addWidget(shear_group)
        layout.addWidget(perspective_group)
        layout.addWidget(wave_group)
        layout.addWidget(affine_group)
        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_matrix_info_tab(self):
        """创建变换矩阵信息显示选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 当前变换类型
        self.transform_type_label = QLabel("当前变换类型：")
        self.transform_type_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.transform_type_label.setStyleSheet("color: #0066cc; padding: 5px;")

        # 变换矩阵显示
        self.matrix_text = QTextEdit()
        self.matrix_text.setReadOnly(True)
        self.matrix_text.setFont(QFont("Courier New", 11))
        self.matrix_text.setMinimumHeight(350)

        # 变换参数显示
        self.params_text = QTextEdit()
        self.params_text.setReadOnly(True)
        self.params_text.setFont(QFont("Microsoft YaHei", 10))
        self.params_text.setMaximumHeight(120)

        btn_refresh = QPushButton("🔄 刷新矩阵信息")
        btn_refresh.clicked.connect(self.update_matrix_info)

        layout.addWidget(self.transform_type_label)
        layout.addWidget(QLabel("变换矩阵："))
        layout.addWidget(self.matrix_text)
        layout.addWidget(QLabel("变换参数："))
        layout.addWidget(self.params_text)
        layout.addWidget(btn_refresh)

        tab.setLayout(layout)
        return tab

    # ------------------ 图像显示辅助函数 ------------------
    def show_image(self, img, label):
        """在指定QLabel上显示OpenCV图像（BGR转RGB）"""
        if img is None:
            return

        # 确保图像是连续数组且数据类型正确
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)

        # BGR 转 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_img)
        # 缩放显示以适应label大小，保持宽高比
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setScaledContents(False)

    def open_image(self):
        """打开图像文件"""
        file, _ = QFileDialog.getOpenFileName(
            self, "打开图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif);;所有文件 (*)"
        )
        if file:
            self.image = cv2.imread(file)
            if self.image is None:
                QMessageBox.warning(self, "错误", "无法加载图像，请检查文件格式。")
                return
            self.result = self.image.copy()
            self.original_size = (self.image.shape[1], self.image.shape[0])

            # 重置变换信息
            self.current_transform = {
                'type': '原始图像',
                'matrix': '单位矩阵',
                'params': {}
            }

            self.show_image(self.image, self.label_src)
            self.show_image(self.result, self.label_dst)
            self.update_matrix_info()
            QMessageBox.information(self, "成功", f"已加载图像: {file}")

    def show_info(self):
        """显示图像信息"""
        if self.image is None:
            QMessageBox.warning(self, "提示", "请先打开图像。")
            return
        h, w, c = self.image.shape
        QMessageBox.information(
            self,
            "图像信息",
            f"📐 尺寸: {w} x {h}\n🎨 通道数: {c}\n📊 数据类型: {self.image.dtype}"
        )

    def reset_image(self):
        """还原到原始图像"""
        if self.image is None:
            QMessageBox.warning(self, "提示", "请先打开图像。")
            return
        self.result = self.image.copy()

        # 重置变换信息
        self.current_transform = {
            'type': '原始图像',
            'matrix': '单位矩阵',
            'params': {}
        }

        self.show_image(self.result, self.label_dst)
        self.update_matrix_info()
        QMessageBox.information(self, "还原", "已还原至原始图像。")

    def save_image(self):
        """保存当前结果图像"""
        if self.result is None:
            QMessageBox.warning(self, "提示", "没有可保存的图像。")
            return
        file, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "",
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg);;所有文件 (*)"
        )
        if file:
            cv2.imwrite(file, self.result)
            QMessageBox.information(self, "保存成功", f"图像已保存至:\n{file}")

    def update_matrix_info(self):
        """更新矩阵信息显示"""
        if self.result is None:
            self.transform_type_label.setText("当前变换类型：未加载图像")
            self.matrix_text.setText("")
            self.params_text.setText("")
            return

        # 更新变换类型
        self.transform_type_label.setText(f"当前变换类型：{self.current_transform['type']}")

        # 显示变换矩阵
        if self.current_transform['matrix'] is not None:
            if isinstance(self.current_transform['matrix'], np.ndarray):
                matrix_str = np.array2string(self.current_transform['matrix'],
                                             precision=4,
                                             suppress_small=True,
                                             separator=', ')
                self.matrix_text.setText(matrix_str)
            else:
                self.matrix_text.setText(str(self.current_transform['matrix']))
        else:
            self.matrix_text.setText("无变换矩阵（非线性变换）")

        # 显示变换参数
        if self.current_transform['params']:
            params_str = ""
            for key, value in self.current_transform['params'].items():
                params_str += f"{key}: {value}\n"
            self.params_text.setText(params_str)
        else:
            self.params_text.setText("无额外参数")

    def set_transform_info(self, transform_type, matrix=None, params=None):
        """设置当前变换信息"""
        self.current_transform['type'] = transform_type
        self.current_transform['matrix'] = matrix
        self.current_transform['params'] = params if params else {}
        self.update_matrix_info()

    # ------------------ 基础空间变换 ------------------
    def zoom_transform(self):
        """缩放变换"""
        if self.result is None:
            QMessageBox.warning(self, "提示", "请先打开图像。")
            return

        factor = self.zoom_factor.value()
        if factor <= 0:
            QMessageBox.warning(self, "错误", "缩放因子必须大于0。")
            return

        h, w = self.result.shape[:2]
        new_w = int(w * factor)
        new_h = int(h * factor)
        if new_w < 1 or new_h < 1:
            QMessageBox.warning(self, "错误", "缩放后尺寸过小。")
            return

        # 构建缩放矩阵
        scale_matrix = np.array([[factor, 0, 0],
                                 [0, factor, 0],
                                 [0, 0, 1]], dtype=np.float32)

        self.result = cv2.resize(self.result, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 记录变换信息
        self.set_transform_info(
            f"缩放变换 (因子: {factor})",
            scale_matrix,
            {'缩放因子': f'{factor}倍', '新尺寸': f'{new_w} x {new_h}'}
        )

        self.show_image(self.result, self.label_dst)

    def rotate_transform(self):
        """旋转变换"""
        if self.result is None:
            QMessageBox.warning(self, "提示", "请先打开图像。")
            return

        angle = self.rotate_angle.value()
        h, w = self.result.shape[:2]
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 构建完整的3x3旋转矩阵
        rot_3x3 = np.eye(3, dtype=np.float32)
        rot_3x3[:2, :] = rotation_matrix

        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        rot_3x3[:2, :] = rotation_matrix

        self.result = cv2.warpAffine(self.result, rotation_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # 记录变换信息
        angle_rad = angle * np.pi / 180
        self.set_transform_info(
            f"旋转变换 (角度: {angle}°)",
            rot_3x3,
            {'旋转角度': f'{angle}°', '弧度': f'{angle_rad:.4f} rad', '新尺寸': f'{new_w} x {new_h}'}
        )

        self.show_image(self.result, self.label_dst)

    def translate_transform(self):
        """平移变换"""
        if self.result is None:
            QMessageBox.warning(self, "提示", "请先打开图像。")
            return

        tx = self.tx.value()
        ty = self.ty.value()
        h, w = self.result.shape[:2]

        # 构建平移矩阵
        translate_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]], dtype=np.float32)

        M = np.float32([[1, 0, tx], [0, 1, ty]])
        new_w = w + abs(tx)
        new_h = h + abs(ty)
        self.result = cv2.warpAffine(self.result, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # 记录变换信息
        self.set_transform_info(
            f"平移变换 (Δx={tx}, Δy={ty})",
            translate_matrix,
            {'x方向平移': f'{tx}像素', 'y方向平移': f'{ty}像素', '新尺寸': f'{new_w} x {new_h}'}
        )

        self.show_image(self.result, self.label_dst)

    def flip_horizontal(self):
        """水平镜像"""
        if self.result is None:
            QMessageBox.warning(self, "提示", "请先打开图像。")
            return

        # 水平镜像矩阵
        flip_matrix = np.array([[-1, 0, self.result.shape[1] - 1],
                                [0, 1, 0],
                                [0, 0, 1]], dtype=np.float32)

        self.result = cv2.flip(self.result, 1)

        self.set_transform_info(
            "水平镜像变换",
            flip_matrix,
            {'变换类型': '水平翻转', '说明': '左右翻转'}
        )

        self.show_image(self.result, self.label_dst)

    def flip_vertical(self):
        """垂直镜像"""
        if self.result is None:
            QMessageBox.warning(self, "提示", "请先打开图像。")
            return

        # 垂直镜像矩阵
        flip_matrix = np.array([[1, 0, 0],
                                [0, -1, self.result.shape[0] - 1],
                                [0, 0, 1]], dtype=np.float32)

        self.result = cv2.flip(self.result, 0)

        self.set_transform_info(
            "垂直镜像变换",
            flip_matrix,
            {'变换类型': '垂直翻转', '说明': '上下翻转'}
        )

        self.show_image(self.result, self.label_dst)

    def shear_transform(self):
        """水平剪切变换"""
        if self.result is None:
            QMessageBox.warning(self, "提示", "请先打开图像。")
            return

        shear = self.shear_factor.value()
        h, w = self.result.shape[:2]

        # 构建水平剪切矩阵
        shear_matrix = np.array([[1, shear, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]], dtype=np.float32)

        M = np.float32([[1, shear, 0], [0, 1, 0]])
        new_w = int(w + abs(shear) * h)
        self.result = cv2.warpAffine(self.result, M, (new_w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        self.set_transform_info(
            f"水平剪切变换 (系数: {shear})",
            shear_matrix,
            {'剪切系数': f'{shear}', '新宽度': f'{new_w}像素'}
        )

        self.show_image(self.result, self.label_dst)

    def shear_v_transform(self):
        """垂直剪切变换"""
        if self.result is None:
            QMessageBox.warning(self, "提示", "请先打开图像。")
            return

        shear = self.shear_v_factor.value()
        h, w = self.result.shape[:2]

        # 构建垂直剪切矩阵
        shear_matrix = np.array([[1, 0, 0],
                                 [shear, 1, 0],
                                 [0, 0, 1]], dtype=np.float32)

        M = np.float32([[1, 0, 0], [shear, 1, 0]])
        new_h = int(h + abs(shear) * w)
        self.result = cv2.warpAffine(self.result, M, (w, new_h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        self.set_transform_info(
            f"垂直剪切变换 (系数: {shear})",
            shear_matrix,
            {'剪切系数': f'{shear}', '新高度': f'{new_h}像素'}
        )

        self.show_image(self.result, self.label_dst)

    # ------------------ 透视变换 ------------------
    def perspective_preset(self):
        """预设的透视变换（梯形效果）"""
        if self.result is None:
            QMessageBox.warning(self, "提示", "请先打开图像。")
            return

        h, w = self.result.shape[:2]

        # 定义原始四个角点
        src_points = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])

        # 定义目标四个角点（产生梯形效果）
        offset_x = w * 0.2
        offset_y = h * 0.1
        dst_points = np.float32([
            [offset_x, offset_y],
            [w - offset_x, offset_y],
            [0, h - 1],
            [w - 1, h - 1]
        ])

        # 计算透视变换矩阵
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # 应用透视变换
        self.result = cv2.warpPerspective(self.result, perspective_matrix, (w, h),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(255, 255, 255))

        # 记录变换信息
        self.set_transform_info(
            "透视变换 (预设梯形效果)",
            perspective_matrix,
            {
                '变换类型': '透视变换',
                '源点': f'{(0, 0), ({w - 1}, 0), (0, {h - 1}), ({w - 1}, {h - 1})}',
                '目标点': f'{dst_points.tolist()}',
                '效果': '梯形变形'
            }
        )

        self.show_image(self.result, self.label_dst)

    def custom_perspective(self):
        """自定义透视变换（简单对话框输入四个点）"""
        if self.result is None:
            QMessageBox.warning(self, "提示", "请先打开图像。")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("自定义透视变换")
        dialog.setModal(True)
        layout = QVBoxLayout()

        layout.addWidget(QLabel("原始图像角点坐标（左上、右上、左下、右下）："))
        layout.addWidget(QLabel("默认使用图像四个角点"))

        layout.addWidget(QLabel("\n目标角点坐标（像素）："))

        # 创建输入框
        points_input = []
        labels = ["左上 (x1, y1):", "右上 (x2, y2):", "左下 (x3, y3):", "右下 (x4, y4):"]
        h, w = self.result.shape[:2]

        # 默认值：产生梯形效果
        default_points = [
            (int(w * 0.2), int(h * 0.1)),
            (int(w * 0.8), int(h * 0.1)),
            (0, h - 1),
            (w - 1, h - 1)
        ]

        for i, label in enumerate(labels):
            h_layout = QHBoxLayout()
            h_layout.addWidget(QLabel(label))
            x_input = QSpinBox()
            x_input.setRange(0, w * 2)
            x_input.setValue(default_points[i][0])
            y_input = QSpinBox()
            y_input.setRange(0, h * 2)
            y_input.setValue(default_points[i][1])
            h_layout.addWidget(x_input)
            h_layout.addWidget(y_input)
            points_input.append((x_input, y_input))
            layout.addLayout(h_layout)

        btn_apply = QPushButton("应用透视变换")
        layout.addWidget(btn_apply)

        dialog.setLayout(layout)

        def apply_perspective():
            # 获取输入点
            src_points = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
            dst_points = np.float32([
                [points_input[0][0].value(), points_input[0][1].value()],
                [points_input[1][0].value(), points_input[1][1].value()],
                [points_input[2][0].value(), points_input[2][1].value()],
                [points_input[3][0].value(), points_input[3][1].value()]
            ])

            # 计算透视变换矩阵
            perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            # 应用变换
            self.result = cv2.warpPerspective(self.result, perspective_matrix, (w, h),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=(255, 255, 255))

            # 记录变换信息
            self.set_transform_info(
                "透视变换 (自定义)",
                perspective_matrix,
                {
                    '变换类型': '透视变换',
                    '源点': f'{(0, 0), ({w - 1}, 0), (0, {h - 1}), ({w - 1}, {h - 1})}',
                    '目标点': f'{dst_points.tolist()}',
                    '说明': '用户自定义四个角点'
                }
            )

            self.show_image(self.result, self.label_dst)
            dialog.accept()

        btn_apply.clicked.connect(apply_perspective)
        dialog.exec_()

    # ------------------ 波浪扭曲特效 ------------------
    def wave_transform_horizontal(self):
        """水平波浪扭曲效果"""
        if self.result is None:
            QMessageBox.warning(self, "提示", "请先打开图像。")
            return

        amplitude = self.wave_amplitude.value()
        frequency = self.wave_frequency.value()

        h, w = self.result.shape[:2]

        # 创建映射网格
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)

        for y in range(h):
            for x in range(w):
                # 水平方向的正弦波偏移
                offset_x = amplitude * math.sin(2 * math.pi * frequency * y)
                map_x[y, x] = x + offset_x
                map_y[y, x] = y

        # 确保坐标在范围内
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)

        # 应用映射
        self.result = cv2.remap(self.result, map_x, map_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # 记录变换信息（非线性变换，无矩阵）
        self.set_transform_info(
            f"水平波浪扭曲 (振幅={amplitude}, 频率={frequency})",
            None,  # 非线性变换，无矩阵表示
            {
                '变换类型': '非线性变换 - 水平波浪',
                '振幅': f'{amplitude}像素',
                '频率': f'{frequency}',
                '变换公式': f"x' = x + {amplitude} * sin(2π * {frequency} * y)"
            }
        )

        self.show_image(self.result, self.label_dst)

    def wave_transform_vertical(self):
        """垂直波浪扭曲效果"""
        if self.result is None:
            QMessageBox.warning(self, "提示", "请先打开图像。")
            return

        amplitude = self.wave_amplitude.value()
        frequency = self.wave_frequency.value()

        h, w = self.result.shape[:2]

        # 创建映射网格
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)

        for y in range(h):
            for x in range(w):
                # 垂直方向的正弦波偏移
                offset_y = amplitude * math.sin(2 * math.pi * frequency * x)
                map_x[y, x] = x
                map_y[y, x] = y + offset_y

        # 确保坐标在范围内
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)

        # 应用映射
        self.result = cv2.remap(self.result, map_x, map_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # 记录变换信息（非线性变换，无矩阵）
        self.set_transform_info(
            f"垂直波浪扭曲 (振幅={amplitude}, 频率={frequency})",
            None,  # 非线性变换，无矩阵表示
            {
                '变换类型': '非线性变换 - 垂直波浪',
                '振幅': f'{amplitude}像素',
                '频率': f'{frequency}',
                '变换公式': f"y' = y + {amplitude} * sin(2π * {frequency} * x)"
            }
        )

        self.show_image(self.result, self.label_dst)

    # ------------------ 自定义仿射变换 ------------------
    def start_affine_selection(self):
        """开始自定义仿射变换（三点选点）"""
        if self.result is None:
            QMessageBox.warning(self, "提示", "请先打开图像。")
            return

        QMessageBox.information(self, "自定义仿射变换",
                                "请在当前图像上选择3个点作为源点，\n"
                                "然后选择对应的3个目标点。\n"
                                "注意：三点不能共线！")

        self.affine_selection_mode = True
        self.src_points = []
        self.dst_points = []
        self.affine_status.setText("状态：请点击图像选择3个源点")

        # 创建一个临时窗口用于选点
        self.affine_dialog = QDialog(self)
        self.affine_dialog.setWindowTitle("自定义仿射变换 - 选点")
        self.affine_dialog.setModal(True)
        layout = QVBoxLayout()

        # 显示当前图像
        img_display = QLabel()
        img_display.setAlignment(Qt.AlignCenter)
        img_display.setMinimumSize(500, 500)

        # 显示当前图像
        if self.result is not None:
            img_rgb = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_display.setPixmap(pixmap)

        self.point_label = QLabel("已选择源点: 0/3")
        self.point_label.setAlignment(Qt.AlignCenter)

        btn_confirm = QPushButton("完成选点并进行仿射变换")
        btn_confirm.clicked.connect(self.apply_affine_transform)
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.affine_dialog.reject)

        layout.addWidget(img_display)
        layout.addWidget(self.point_label)
        layout.addWidget(btn_confirm)
        layout.addWidget(btn_cancel)

        self.affine_dialog.setLayout(layout)

        # 实现简单的鼠标点击事件
        img_display.mousePressEvent = self.on_affine_click
        self.current_img_display = img_display

        self.affine_dialog.exec_()
        self.affine_selection_mode = False
        self.affine_status.setText("状态：未开始")

    def on_affine_click(self, event):
        """处理仿射变换选点"""
        if not self.affine_selection_mode:
            return

        # 获取点击位置（相对于图像的坐标）
        pos = event.pos()
        label_size = self.current_img_display.size()
        pixmap = self.current_img_display.pixmap()

        if pixmap is None:
            return

        # 计算实际图像坐标
        img_h, img_w = self.result.shape[:2]
        display_w = pixmap.width()
        display_h = pixmap.height()

        # 计算偏移和缩放
        x_offset = (label_size.width() - display_w) // 2
        y_offset = (label_size.height() - display_h) // 2

        x_img = int((pos.x() - x_offset) * img_w / display_w)
        y_img = int((pos.y() - y_offset) * img_h / display_h)

        if 0 <= x_img < img_w and 0 <= y_img < img_h:
            if len(self.src_points) < 3:
                self.src_points.append((x_img, y_img))
                self.point_label.setText(f"已选择源点: {len(self.src_points)}/3")

                # 在图像上标记点（简化实现）
                self.current_img_display.setPixmap(self.draw_points_on_image())

                if len(self.src_points) == 3:
                    # 选择目标点
                    self.point_label.setText("源点选择完成，请选择3个目标点")
                    self.dst_points = []
                    QMessageBox.information(self, "选择目标点",
                                            "请继续选择3个目标点（变换后的位置）")
            elif len(self.dst_points) < 3:
                self.dst_points.append((x_img, y_img))
                self.point_label.setText(f"已选择目标点: {len(self.dst_points)}/3")

                # 在图像上标记点
                self.current_img_display.setPixmap(self.draw_points_on_image())

                if len(self.dst_points) == 3:
                    self.apply_affine_transform()

    def draw_points_on_image(self):
        """在图像上绘制选中的点"""
        if self.result is None:
            return None

        img_copy = self.result.copy()
        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

        # 绘制源点（红色）
        for i, (x, y) in enumerate(self.src_points):
            cv2.circle(img_rgb, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img_rgb, str(i + 1), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 绘制目标点（绿色）
        for i, (x, y) in enumerate(self.dst_points):
            cv2.circle(img_rgb, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(img_rgb, str(i + 1), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        return pixmap

    def apply_affine_transform(self):
        """应用自定义仿射变换"""
        if len(self.src_points) == 3 and len(self.dst_points) == 3:
            src = np.float32(self.src_points)
            dst = np.float32(self.dst_points)

            # 计算仿射变换矩阵
            affine_matrix = cv2.getAffineTransform(src, dst)

            # 构建完整的3x3矩阵
            affine_3x3 = np.eye(3, dtype=np.float32)
            affine_3x3[:2, :] = affine_matrix

            h, w = self.result.shape[:2]
            # 计算新图像尺寸
            points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            transformed = cv2.transform(points.reshape(-1, 1, 2), affine_matrix)
            new_w = int(np.max(transformed[:, 0, 0]) - np.min(transformed[:, 0, 0]))
            new_h = int(np.max(transformed[:, 0, 1]) - np.min(transformed[:, 0, 1]))

            # 应用仿射变换
            self.result = cv2.warpAffine(self.result, affine_matrix, (new_w, new_h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(255, 255, 255))

            # 记录变换信息
            self.set_transform_info(
                "自定义仿射变换",
                affine_3x3,
                {
                    '源点': f'{self.src_points}',
                    '目标点': f'{self.dst_points}',
                    '新尺寸': f'{new_w} x {new_h}'
                }
            )

            self.show_image(self.result, self.label_dst)
            self.affine_dialog.accept()
        else:
            QMessageBox.warning(self, "错误", f"请选择完整的3个源点({len(self.src_points)}/3)和3个目标点({len(self.dst_points)}/3)！")

    # ------------------ 窗口大小适应 ------------------
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放显示图像"""
        if self.image is not None:
            self.show_image(self.image, self.label_src)
        if self.result is not None:
            self.show_image(self.result, self.label_dst)
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageTransformDemo()
    window.show()
    sys.exit(app.exec_())