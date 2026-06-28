# -*- coding: utf-8 -*-
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.image_processor import ImageProcessor


class ImageService:

    def __init__(self):
        self.processor = ImageProcessor()
        self.image: np.ndarray | None = None     # 原始图，永不修改
        self.result: np.ndarray | None = None    # 当前空间域处理结果（累积）
        self.spectrum: np.ndarray | None = None  # result 的频谱（自动更新）

    def _update_spectrum(self):
        if self.result is not None:
            self.spectrum = self.processor.compute_spectrum(self.result)

    def load_image(self, path: str):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        self.image = img
        self.result = img.copy()
        self._update_spectrum()
        return self.result, self.spectrum

    def reset(self):
        if self.image is None:
            return None
        self.result = self.image.copy()
        self._update_spectrum()
        return self.result, self.spectrum

    def get_original_spectrum(self):
        if self.image is None:
            return None
        return self.processor.compute_spectrum(self.image)

    # ── 空间域变换（累积，更新 result 和 spectrum）────────────────────────────

    def zoom(self, factor: float):
        self.result = self.processor.zoom(self.result, factor)
        self._update_spectrum()
        return self.result, self.spectrum

    def translate(self, tx: int, ty: int):
        self.result = self.processor.translate(self.result, tx, ty)
        self._update_spectrum()
        return self.result, self.spectrum

    def rotate(self, angle: float):
        self.result = self.processor.rotate(self.result, angle)
        self._update_spectrum()
        return self.result, self.spectrum

    def gaussian_smooth(self, ksize: int, sigma: float):
        self.result = self.processor.gaussian_smooth(self.result, ksize, sigma)
        self._update_spectrum()
        return self.result, self.spectrum

    def laplacian_sharpen(self):
        self.result = self.processor.laplacian_sharpen(self.result)
        self._update_spectrum()
        return self.result, self.spectrum

    def median_denoise(self, ksize: int):
        self.result = self.processor.median_denoise(self.result, ksize)
        self._update_spectrum()
        return self.result, self.spectrum

    # ── 频率域滤波（不修改 result，返回临时结果）─────────────────────────────

    def butterworth_lowpass(self, cutoff: int, order: int):
        filtered = self.processor.butterworth_lowpass(self.result, cutoff, order)
        spectrum = self.processor.compute_spectrum(filtered)
        return filtered, spectrum

    def butterworth_highpass(self, cutoff: int, order: int):
        filtered = self.processor.butterworth_highpass(self.result, cutoff, order)
        spectrum = self.processor.compute_spectrum(filtered)
        return filtered, spectrum
