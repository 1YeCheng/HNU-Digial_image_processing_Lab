import cv2
import numpy as np
from core.image_processor import ImageProcessor


class ImageService:

    def __init__(self):
        self.processor = ImageProcessor()
        self.image = None    # 原始图像，始终不变
        self.result = None   # 当前显示结果（基础变换累积）

    def load_image(self, path):
        self.image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.image is None:
            return None
        self.result = self.image.copy()
        return self.result

    def reset(self):
        if self.image is None:
            return None
        self.result = self.image.copy()
        return self.result

    # ===== 基础变换（基于 result 累积）=====

    def gray(self):
        self.result = self.processor.gray(self.result)
        return self.result

    def binary(self):
        self.result = self.processor.binary(self.result)
        return self.result

    def inverse(self):
        self.result = self.processor.inverse(self.result)
        return self.result

    def gamma(self):
        self.result = self.processor.gamma(self.result)
        return self.result

    def log_transform(self):
        self.result = self.processor.log_transform(self.result)
        return self.result

    def exp_transform(self):
        self.result = self.processor.exp_transform(self.result)
        return self.result

    def resize_half(self):
        self.result = self.processor.resize_half(self.result)
        return self.result

    def get_info(self):
        if self.result is None:
            return None
        h, w, c = self.result.shape
        return w, h, c

    # ===== 特效（始终基于原始图像，不累积）=====

    def glass(self):
        if self.image is None:
            return None
        return self.processor.glass(self.image)

    def relief(self):
        if self.image is None:
            return None
        return self.processor.relief(self.image)

    def oil(self):
        if self.image is None:
            return None
        return self.processor.oil(self.image)

    def mask(self):
        if self.image is None:
            return None
        return self.processor.mask(self.image)

    def sketch(self):
        if self.image is None:
            return None
        return self.processor.sketch(self.image)

    def old(self):
        if self.image is None:
            return None
        return self.processor.old(self.image)

    def lighting(self):
        if self.image is None:
            return None
        return self.processor.lighting(self.image)

    def cartoonize(self):
        if self.image is None:
            return None
        return self.processor.cartoonize(self.image)
