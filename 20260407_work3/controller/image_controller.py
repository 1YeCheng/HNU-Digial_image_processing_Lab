# -*- coding: utf-8 -*-
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.image_service import ImageService


class ImageController:

    def __init__(self):
        self.service = ImageService()

    def open_image(self, path: str):
        return self.service.load_image(path)

    def reset(self):
        return self.service.reset()

    def get_original_spectrum(self):
        return self.service.get_original_spectrum()

    # ── 空间域变换 ────────────────────────────────────────────────────────────

    def zoom(self, factor: float):
        return self.service.zoom(factor)

    def translate(self, tx: int, ty: int):
        return self.service.translate(tx, ty)

    def rotate(self, angle: float):
        return self.service.rotate(angle)

    def gaussian_smooth(self, ksize: int, sigma: float):
        return self.service.gaussian_smooth(ksize, sigma)

    def laplacian_sharpen(self):
        return self.service.laplacian_sharpen()

    def median_denoise(self, ksize: int):
        return self.service.median_denoise(ksize)

    # ── 频率域滤波 ────────────────────────────────────────────────────────────

    def butterworth_lowpass(self, cutoff: int, order: int):
        return self.service.butterworth_lowpass(cutoff, order)

    def butterworth_highpass(self, cutoff: int, order: int):
        return self.service.butterworth_highpass(cutoff, order)
