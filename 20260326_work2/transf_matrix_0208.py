# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class ImageTransformDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Transform & Stitching Tool v2.1")
        self.resize(1600, 900)
        self.setMinimumSize(1200, 700)

        self.image = None  # Original image
        self.result = None  # Current result image

        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left: Source Image Panel
        src_group = QGroupBox("Source Image (Original)")
        src_layout = QVBoxLayout()
        self.label_src = QLabel("No Image")
        self.label_src.setAlignment(Qt.AlignCenter)
        self.label_src.setStyleSheet("border: 1px solid #ccc; background: #333;")
        src_layout.addWidget(self.label_src)
        src_group.setLayout(src_layout)

        # Middle: Result Image Panel
        dst_group = QGroupBox("Transformation Result (Current Canvas)")
        dst_layout = QVBoxLayout()
        self.label_dst = QLabel("No Result")
        self.label_dst.setAlignment(Qt.AlignCenter)
        self.label_dst.setStyleSheet("border: 1px solid #ccc; background: #333;")
        dst_layout.addWidget(self.label_dst)
        dst_group.setLayout(dst_layout)

        # Right: Control Panel
        control_tabs = QTabWidget()
        control_tabs.addTab(self.file_tab(), "File")
        control_tabs.addTab(self.transform_tab(), "Transform")
        control_tabs.addTab(self.stitch_tab(), "Stitch")
        control_tabs.setMaximumWidth(350)

        main_layout.addWidget(src_group, 4)
        main_layout.addWidget(dst_group, 4)
        main_layout.addWidget(control_tabs, 2)

    def file_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        btn_open = QPushButton("📂 Open Image");
        btn_open.clicked.connect(self.open_image)
        btn_reset = QPushButton("🔄 Reset Result");
        btn_reset.clicked.connect(self.reset_image)
        layout.addWidget(btn_open);
        layout.addWidget(btn_reset);
        layout.addStretch()
        return tab

    def transform_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Scaling Section
        zoom_group = QGroupBox("Physical Scaling")
        zoom_layout = QVBoxLayout()
        self.zoom_factor = QDoubleSpinBox()
        self.zoom_factor.setRange(0.1, 5.0);
        self.zoom_factor.setValue(0.5);
        self.zoom_factor.setSuffix("x")
        btn_zoom = QPushButton("Execute Scale");
        btn_zoom.clicked.connect(self.zoom_transform)
        zoom_layout.addWidget(QLabel("Factor (< 1.0 to shrink):"))
        zoom_layout.addWidget(self.zoom_factor);
        zoom_layout.addWidget(btn_zoom)
        zoom_group.setLayout(zoom_layout)

        # Rotation Section
        rot_group = QGroupBox("Rotate")
        rot_layout = QVBoxLayout()
        self.rot_angle = QSpinBox()
        self.rot_angle.setRange(-360, 360);
        self.rot_angle.setValue(45)
        btn_rot = QPushButton("Rotate Image");
        btn_rot.clicked.connect(self.rotate_transform)
        rot_layout.addWidget(self.rot_angle);
        rot_layout.addWidget(btn_rot)
        rot_group.setLayout(rot_layout)

        layout.addWidget(zoom_group);
        layout.addWidget(rot_group);
        layout.addStretch()
        return tab

    def stitch_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Merge current result with a new image:"))
        btn_smart = QPushButton("🚀 Smart Stitch (Panorama)");
        btn_smart.clicked.connect(self.smart_stitch)
        btn_basic = QPushButton("🔗 Basic Side-by-Side Join");
        btn_basic.clicked.connect(self.basic_join)
        layout.addWidget(btn_smart);
        layout.addWidget(btn_basic);
        layout.addStretch()
        return tab

    # --- Functions ---

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Images (*.jpg *.png)")
        if path:
            self.image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.result = self.image.copy()
            self.show_image(self.image, self.label_src)
            self.show_image(self.result, self.label_dst)

    def show_image(self, img, label):
        if img is None: return
        # Info overlay: show actual pixel size
        h, w = img.shape[:2]
        display_img = img.copy()
        cv2.putText(display_img, f"{w}x{h} px", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def zoom_transform(self):
        """ Shrink or enlarge the physical pixel size of the result """
        if self.result is None: return
        factor = self.zoom_factor.value()
        h, w = self.result.shape[:2]
        new_w, new_h = int(w * factor), int(h * factor)

        # Avoid zero size
        new_w = max(new_w, 1);
        new_h = max(new_h, 1)

        # Physical resize
        interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
        self.result = cv2.resize(self.result, (new_w, new_h), interpolation=interp)
        self.show_image(self.result, self.label_dst)

    def rotate_transform(self):
        if self.result is None: return
        angle = self.rot_angle.value()
        h, w = self.result.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate bound box
        cos = np.abs(M[0, 0]);
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos));
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - center[0];
        M[1, 2] += (nH / 2) - center[1]

        self.result = cv2.warpAffine(self.result, M, (nW, nH), borderValue=(0, 0, 0))
        self.show_image(self.result, self.label_dst)

    def smart_stitch(self):
        """ Intelligent panorama stitching """
        if self.result is None: return
        path, _ = QFileDialog.getOpenFileName(self, "Select overlapping image", "", "Images (*.jpg *.png)")
        if not path: return

        img2 = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        stitcher = cv2.Stitcher_create()
        status, stitched = stitcher.stitch([self.result, img2])

        if status == cv2.Stitcher_OK:
            self.result = stitched
            self.show_image(self.result, self.label_dst)
        else:
            msg = QMessageBox.question(self, "Stitching Failed",
                                       f"Err:{status}. Smart stitching failed (need more overlap).\n"
                                       "Would you like to perform a Basic Side-by-Side Join instead?",
                                       QMessageBox.Yes | QMessageBox.No)
            if msg == QMessageBox.Yes:
                self.perform_basic_join(img2)

    def basic_join(self):
        """ Manual side-by-side concatenation """
        if self.result is None: return
        path, _ = QFileDialog.getOpenFileName(self, "Select any image", "", "Images (*.jpg *.png)")
        if not path: return
        img2 = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.perform_basic_join(img2)

    def perform_basic_join(self, img2):
        """ Resize second image to match height of first and concatenate """
        h1, w1 = self.result.shape[:2]
        h2, w2 = img2.shape[:2]

        # Resize img2 to match height of self.result
        scale = h1 / h2
        img2_resized = cv2.resize(img2, (int(w2 * scale), h1), interpolation=cv2.INTER_LINEAR)

        # Concatenate horizontally
        self.result = cv2.hconcat([self.result, img2_resized])
        self.show_image(self.result, self.label_dst)

    def reset_image(self):
        if self.image is not None:
            self.result = self.image.copy()
            self.show_image(self.result, self.label_dst)

    def resizeEvent(self, event):
        self.show_image(self.image, self.label_src)
        self.show_image(self.result, self.label_dst)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ImageTransformDemo()
    win.show()
    sys.exit(app.exec_())