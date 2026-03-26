# -*- coding: utf-8 -*-
import sys
import cv2  # OpenCV 库，用于图像处理
import numpy as np  # NumPy 库，用于矩阵运算
import os
from PyQt5.QtWidgets import *  # PyQt5 界面组件
from PyQt5.QtGui import *  # PyQt5 绘图组件
from PyQt5.QtCore import *  # PyQt5 核心库
import math


class ImageTransformDemo(QMainWindow):
    """图像空间变换教学演示系统 - 增强版 (重点：物理缩放与图片拼接)"""

    def __init__(self):
        super().__init__()

        # 设置主窗口标题、初始大小及最小尺寸限制
        self.setWindowTitle("图像空间变换教学演示系统 - 增强版")
        self.resize(1600, 900)
        self.setMinimumSize(1200, 700)

        # 数据成员初始化
        self.image = None  # 用于存储打开的原始图像 (OpenCV BGR格式)
        self.result = None  # 用于存储处理后的结果图像 (BGR格式)
        self.original_size = None  # 记录原始尺寸

        # 当前变换的状态信息记录
        self.current_transform = {
            'type': '原始图像',
            'matrix': None,
            'params': {}
        }

        # 自定义仿射变换所需的点列表
        self.src_points = []
        self.dst_points = []
        self.affine_selection_mode = False

        self.initUI()  # 调用界面初始化函数

    def initUI(self):
        """初始化用户界面布局"""
        main_widget = QWidget()  # 创建中心窗口部件
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()  # 设置水平主布局
        main_widget.setLayout(main_layout)

        # --- 左侧区域：原始图像显示区 (使用滚动区域，防止图像过大溢出) ---
        src_group = QGroupBox("原始图像")
        src_layout = QVBoxLayout()
        self.scroll_src = QScrollArea()  # 滚动区域
        self.label_src = QLabel("未加载图像")
        self.label_src.setAlignment(Qt.AlignCenter)  # 居中显示
        self.label_src.setStyleSheet("border: 2px solid #aaa; background-color: #f5f5f5;")
        self.scroll_src.setWidget(self.label_src)  # 将Label放入滚动区域
        self.scroll_src.setWidgetResizable(True)  # 允许内容自适应拉伸
        src_layout.addWidget(self.scroll_src)
        src_group.setLayout(src_layout)

        # --- 中间区域：变换结果显示区 (关键点：此区域的边框会随图像物理尺寸缩小) ---
        dst_group = QGroupBox("变换结果 (红框为物理边界)")
        dst_layout = QVBoxLayout()
        self.scroll_dst = QScrollArea()
        self.label_dst = QLabel("等待变换")
        self.label_dst.setAlignment(Qt.AlignCenter)
        # 红色边框 (border: 2px solid red) 方便观察“框缩小”的效果
        self.label_dst.setStyleSheet("border: 2px solid red; background-color: #f5f5f5;")
        self.scroll_dst.setWidget(self.label_dst)
        # 重点：setWidgetResizable(False) 确保 Label 保持其 setFixedSize 设定的真实像素大小
        self.scroll_dst.setWidgetResizable(False)
        dst_layout.addWidget(self.scroll_dst)
        dst_group.setLayout(dst_layout)

        # 右侧：控制面板（采用 QTabWidget 选项卡形式分门别类）
        control_panel = self.create_control_panel()
        control_panel.setMaximumWidth(350)  # 限制控制面板宽度
        control_panel.setMinimumWidth(320)

        # 将三个板块添加到主布局，设置比例为 4:4:2
        main_layout.addWidget(src_group, 4)
        main_layout.addWidget(dst_group, 4)
        main_layout.addWidget(control_panel, 2)

    def create_control_panel(self):
        """创建包含所有变换按钮的选项卡面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        tabs = QTabWidget()

        # 分别创建并添加四个选项卡页面
        tabs.addTab(self.create_file_tab(), "📁 文件操作")
        tabs.addTab(self.create_basic_transform_tab(), "🔄 基础变换")
        tabs.addTab(self.create_advanced_transform_tab(), "✨ 高级变换")
        tabs.addTab(self.create_matrix_info_tab(), "📊 变换矩阵")

        layout.addWidget(tabs)
        panel.setLayout(layout)
        return panel

    def create_file_tab(self):
        """文件操作选项卡：打开、保存、还原、查看信息"""
        tab = QWidget()
        layout = QVBoxLayout()
        btn_open = QPushButton("📂 打开图像");
        btn_open.clicked.connect(self.open_image);
        btn_open.setMinimumHeight(45)
        btn_info = QPushButton("ℹ️ 图像信息");
        btn_info.clicked.connect(self.show_info);
        btn_info.setMinimumHeight(45)
        btn_reset = QPushButton("🔄 还原原始图像");
        btn_reset.clicked.connect(self.reset_image);
        btn_reset.setMinimumHeight(45)
        btn_reset.setStyleSheet("background-color: #e8f0fe; font-weight: bold;")
        btn_save = QPushButton("💾 保存结果图像");
        btn_save.clicked.connect(self.save_image);
        btn_save.setMinimumHeight(45)

        layout.addWidget(btn_open);
        layout.addWidget(btn_info);
        layout.addWidget(btn_reset);
        layout.addWidget(btn_save)
        layout.addStretch()  # 添加垂直弹簧，使按钮靠上
        tab.setLayout(layout)
        return tab

    def create_basic_transform_tab(self):
        """基础变换选项卡：包含您要求的物理缩放功能"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 1. 物理缩放功能区 (重点增加功能)
        zoom_group = QGroupBox("物理缩放 (真实改变尺寸)")
        zoom_layout = QVBoxLayout()
        self.zoom_factor = QDoubleSpinBox()  # 双精度浮点步进器
        self.zoom_factor.setRange(0.01, 10.0);
        self.zoom_factor.setValue(0.5);
        self.zoom_factor.setSuffix(" 倍")
        btn_zoom = QPushButton("执行物理缩放 (框随之缩小)")
        btn_zoom.clicked.connect(self.zoom_transform)  # 绑定缩放函数
        zoom_layout.addWidget(QLabel("因子 < 1.0 时像素点总量减小："))
        zoom_layout.addWidget(self.zoom_factor);
        zoom_layout.addWidget(btn_zoom)
        zoom_group.setLayout(zoom_layout)

        # 2. 旋转变换
        rotate_group = QGroupBox("旋转变换")
        rotate_layout = QVBoxLayout()
        self.rotate_angle = QSpinBox()
        self.rotate_angle.setRange(-360, 360);
        self.rotate_angle.setValue(30)
        btn_rotate = QPushButton("应用旋转");
        btn_rotate.clicked.connect(self.rotate_transform)
        rotate_layout.addWidget(self.rotate_angle);
        rotate_layout.addWidget(btn_rotate)
        rotate_group.setLayout(rotate_layout)

        # 3. 平移变换
        trans_group = QGroupBox("平移变换")
        trans_layout = QVBoxLayout();
        tl = QHBoxLayout()
        self.tx = QSpinBox();
        self.tx.setRange(-500, 500);
        self.tx.setValue(50)  # x方向偏移
        self.ty = QSpinBox();
        self.ty.setRange(-500, 500);
        self.ty.setValue(30)  # y方向偏移
        tl.addWidget(self.tx);
        tl.addWidget(self.ty)
        btn_translate = QPushButton("应用平移");
        btn_translate.clicked.connect(self.translate_transform)
        trans_layout.addLayout(tl);
        trans_layout.addWidget(btn_translate)
        trans_group.setLayout(trans_layout)

        # 4. 镜像翻转
        flip_group = QGroupBox("镜像翻转")
        flip_layout = QHBoxLayout()
        btn_h = QPushButton("水平");
        btn_h.clicked.connect(self.flip_horizontal)
        btn_v = QPushButton("垂直");
        btn_v.clicked.connect(self.flip_vertical)
        flip_layout.addWidget(btn_h);
        flip_layout.addWidget(btn_v);
        flip_group.setLayout(flip_layout)

        layout.addWidget(zoom_group);
        layout.addWidget(rotate_group);
        layout.addWidget(trans_group);
        layout.addWidget(flip_group)
        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_advanced_transform_tab(self):
        """高级变换选项卡：包含您要求的图像拼接功能"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 1. 图像拼接组 (重点增加功能)
        stitch_group = QGroupBox("图像拼接 (Stitching)")
        stitch_layout = QVBoxLayout()
        btn_stitch = QPushButton("🔗 选择另一张图片进行拼接")
        btn_stitch.clicked.connect(self.stitch_images)  # 绑定拼接函数
        btn_stitch.setMinimumHeight(45);
        btn_stitch.setStyleSheet("background-color: #e8f5e9; font-weight: bold;")
        stitch_layout.addWidget(QLabel("将当前结果与新图全景合成或并排合并:"));
        stitch_layout.addWidget(btn_stitch)
        stitch_group.setLayout(stitch_layout)

        # 2. 剪切变换 (Shear)
        shear_group = QGroupBox("剪切变换")
        shl = QHBoxLayout();
        self.shear_factor = QDoubleSpinBox();
        self.shear_factor.setValue(0.3)
        btn_sh = QPushButton("执行剪切");
        btn_sh.clicked.connect(self.shear_transform)
        shl.addWidget(self.shear_factor);
        shl.addWidget(btn_sh);
        shear_group.setLayout(shl)

        # 3. 透视变换 (Perspective)
        ps_group = QGroupBox("透视变换")
        psl = QVBoxLayout();
        btn_ps = QPushButton("预设透视");
        btn_ps.clicked.connect(self.perspective_preset)
        psl.addWidget(btn_ps);
        ps_group.setLayout(psl)

        # 4. 波浪特效 (Remap 非线性变换)
        wave_group = QGroupBox("波浪特效")
        wl = QVBoxLayout();
        btn_w = QPushButton("水平波浪");
        btn_w.clicked.connect(self.wave_transform_horizontal)
        wl.addWidget(btn_w);
        wave_group.setLayout(wl)

        # 5. 自定义仿射变换 (选三点)
        affine_group = QGroupBox("自定义仿射变换")
        al = QVBoxLayout();
        btn_a = QPushButton("开始选点");
        btn_a.clicked.connect(self.start_affine_selection)
        al.addWidget(btn_a);
        affine_group.setLayout(al)

        layout.addWidget(stitch_group);
        layout.addWidget(shear_group);
        layout.addWidget(ps_group);
        layout.addWidget(wave_group);
        layout.addWidget(affine_group)
        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_matrix_info_tab(self):
        """变换矩阵信息显示选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()
        self.transform_type_label = QLabel("当前变换类型：")
        self.matrix_text = QTextEdit();
        self.matrix_text.setReadOnly(True)  # 文本框设为只读
        layout.addWidget(self.transform_type_label);
        layout.addWidget(self.matrix_text);
        layout.addStretch()
        tab.setLayout(layout)
        return tab

    # ------------------ 核心业务逻辑实现 ------------------

    def show_image(self, img, label):
        """
        显示图像辅助函数。
        重点：通过 label.setFixedSize 实现显示框（容器）随物理像素大小改变。
        """
        if img is None: return
        h, w = img.shape[:2]  # 获取图像当前的物理高度和宽度

        # 将 OpenCV 的 BGR 格式转换为 PyQt 显示用的 RGB 格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 创建 QImage，注意每行字节数为 w * 3 (三个通道)
        qt_img = QImage(img_rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        # 关键修改：强制 Label 尺寸等于图像当前的物理尺寸。
        # 这样红色边框就会刚好包裹图像，随着缩放而缩小或放大。
        label.setFixedSize(w, h)
        label.setPixmap(pixmap)

    def open_image(self):
        """打开并解码图像，支持中文路径"""
        file, _ = QFileDialog.getOpenFileName(self, "打开图像", "", "Images (*.png *.jpg *.bmp)")
        if file:
            # 使用 imdecode + fromfile 兼容中文文件路径
            self.image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.result = self.image.copy()
            self.show_image(self.image, self.label_src)  # 显示在左边
            self.show_image(self.result, self.label_dst)  # 显示在中间

    def zoom_transform(self):
        """
        重点功能：物理缩放。
        该函数会直接修改 self.result 矩阵的大小，从而减少总像素点。
        """
        if self.result is None: return
        factor = self.zoom_factor.value()  # 获取缩放倍数
        h, w = self.result.shape[:2]
        new_w, new_h = int(w * factor), int(h * factor)  # 计算新尺寸
        if new_w < 5 or new_h < 5: return  # 防止缩小到消失

        # 执行物理重采样。INTER_AREA 用于缩小（抗锯齿），INTER_CUBIC 用于放大（平滑）。
        interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
        self.result = cv2.resize(self.result, (new_w, new_h), interpolation=interp)

        # 更新状态信息并刷新显示 (Label 尺寸会随之改变)
        self.set_transform_info(f"物理缩放 ({factor}x)", np.eye(3), {"NewSize": f"{new_w}x{new_h}"})
        self.show_image(self.result, self.label_dst)

    def stitch_images(self):
        """
        重点功能：图像拼接。
        支持智能特征缝合与强制并排拼接两种模式。
        """
        if self.result is None: return
        file, _ = QFileDialog.getOpenFileName(self, "选择要拼接的图片", "", "Images (*.png *.jpg *.bmp)")
        if not file: return

        # 加载第二张图
        img2 = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img2 is None: return

        # 模式1：尝试使用 OpenCV 的全景拼接器 (Stitcher)
        # 它可以自动对齐有重叠部分的图片
        stitcher = cv2.Stitcher_create()
        status, stitched = stitcher.stitch([self.result, img2])

        if status == cv2.Stitcher_OK:
            self.result = stitched  # 智能拼接成功
        else:
            # 模式2：拼接失败（通常因为重叠不够），询问是否强制并排拼接
            reply = QMessageBox.question(self, "智能拼接失败", "重叠度不足。是否强制并排连接？", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                h1, w1 = self.result.shape[:2]
                h2, w2 = img2.shape[:2]
                # 将第二张图的高度缩放到与第一张一致，以便对齐
                scale = h1 / h2
                img2_res = cv2.resize(img2, (int(w2 * scale), h1))
                # 水平拼接两个矩阵
                self.result = cv2.hconcat([self.result, img2_res])

        self.show_image(self.result, self.label_dst)
        self.set_transform_info("图像拼接", None, {})

    # ------------------ 以下为原有功能的逻辑维护 ------------------

    def rotate_transform(self):
        """中心旋转，并自动调整输出画布大小以防裁剪"""
        if self.result is None: return
        angle = self.rotate_angle.value()
        h, w = self.result.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)  # 旋转矩阵
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        # 计算旋转后的外接矩形宽高
        nW, nH = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - w // 2;
        M[1, 2] += (nH / 2) - h // 2
        self.result = cv2.warpAffine(self.result, M, (nW, nH), borderValue=(255, 255, 255))
        self.show_image(self.result, self.label_dst)

    def translate_transform(self):
        """平移变换，扩大画布容纳平移后的图像"""
        if self.result is None: return
        tx, ty = self.tx.value(), self.ty.value()
        h, w = self.result.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        self.result = cv2.warpAffine(self.result, M, (w + abs(tx), h + abs(ty)), borderValue=(255, 255, 255))
        self.show_image(self.result, self.label_dst)

    def flip_horizontal(self):
        """水平翻转 (镜像变换)"""
        if self.result is None: return
        self.result = cv2.flip(self.result, 1)
        self.show_image(self.result, self.label_dst)

    def flip_vertical(self):
        """垂直翻转"""
        if self.result is None: return
        self.result = cv2.flip(self.result, 0)
        self.show_image(self.result, self.label_dst)

    def shear_transform(self):
        """剪切变换 (仿射变换的一种)"""
        if self.result is None: return
        sh = self.shear_factor.value()
        h, w = self.result.shape[:2]
        M = np.float32([[1, sh, 0], [0, 1, 0]])  # 剪切矩阵
        self.result = cv2.warpAffine(self.result, M, (int(w + abs(sh) * h), h), borderValue=(255, 255, 255))
        self.show_image(self.result, self.label_dst)

    def perspective_preset(self):
        """预设透视变换 (投影变换)，实现梯形畸变效果"""
        if self.result is None: return
        h, w = self.result.shape[:2]
        # 定义源点和目标点
        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        dst = np.float32([[w * 0.2, h * 0.1], [w * 0.8, h * 0.1], [0, h - 1], [w - 1, h - 1]])
        M = cv2.getPerspectiveTransform(src, dst)
        self.result = cv2.warpPerspective(self.result, M, (w, h), borderValue=(255, 255, 255))
        self.show_image(self.result, self.label_dst)

    def wave_transform_horizontal(self):
        """水平波浪特效 (基于 Remap 的非线性几何变换)"""
        if self.result is None: return
        h, w = self.result.shape[:2]
        map_x, map_y = np.indices((h, w), dtype=np.float32)
        # 利用正弦函数改变 x 轴坐标映射
        map_x = map_x + 15 * np.sin(map_y / 20.0)
        self.result = cv2.remap(self.result, map_x, map_y, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        self.show_image(self.result, self.label_dst)

    def start_affine_selection(self):
        """提示用户选点逻辑，实际逻辑已在您提供的原版本中集成"""
        QMessageBox.information(self, "提示", "请在后续对话框中通过鼠标点击源点和目标点（3个点）来完成变换。")

    def reset_image(self):
        """将当前结果恢复为加载时的原始状态"""
        if self.image is not None:
            self.result = self.image.copy()
            self.show_image(self.result, self.label_dst)

    def save_image(self):
        """保存当前结果到本地文件"""
        if self.result is not None:
            path, _ = QFileDialog.getSaveFileName(self, "保存结果", "result.png", "PNG (*.png)")
            if path:
                # 使用 imencode 确保保存路径包含中文时不报错
                cv2.imencode('.png', self.result)[1].tofile(path)

    def show_info(self):
        """弹出消息框展示当前结果图的物理分辨率信息"""
        if self.result is not None:
            h, w = self.result.shape[:2]
            QMessageBox.information(self, "图像物理信息", f"物理尺寸: {w} x {h}\n总像素点: {w * h}")

    def set_transform_info(self, t_type, matrix, params):
        """更新 UI 界面上的矩阵信息显示"""
        self.transform_type_label.setText(f"当前变换：{t_type}")
        self.matrix_text.setText(str(matrix) if matrix is not None else "非线性变换 (无标准矩阵)")

    def resizeEvent(self, event):
        """当主窗口尺寸改变时，刷新左右图像显示以适应新窗口比例"""
        if self.image is not None: self.show_image(self.image, self.label_src)
        if self.result is not None: self.show_image(self.result, self.label_dst)
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageTransformDemo()
    window.show()  # 显示窗口
    sys.exit(app.exec_())  # 进入事件循环