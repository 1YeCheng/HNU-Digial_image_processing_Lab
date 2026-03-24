import cv2
import numpy as np
import math


class ImageProcessor:

    def gray(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def binary(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def inverse(self, img):

        return 255 - img

    def gamma(self, img, gamma=0.5):

        img = img / 255.0
        img = np.power(img, gamma)

        return np.uint8(img * 255)

    def log_transform(self, img):

        img = img.astype(np.float32)

        img = np.log1p(img)

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        return np.uint8(img)

    def exp_transform(self, img):

        img = img.astype(np.float32) / 255.0

        img = np.exp(img)

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        return np.uint8(img)

    def resize_half(self, img):

        return cv2.resize(img, None, fx=0.5, fy=0.5)

    def window_level_transform(self, img, window_width, window_level):

        """
        模拟CT图像窗宽窗位变换

        参数
        ----------
        img : numpy array
            输入图像 (BGR)

        window_width : int
            窗宽

        window_level : int
            窗位

        返回
        ----------
        处理后的图像
        """

        # 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        min_val = window_level - window_width / 2
        max_val = window_level + window_width / 2

        result = np.zeros_like(gray, dtype=np.float32)

        # 分段处理
        result[gray <= min_val] = 0
        result[gray >= max_val] = 255

        mask = (gray > min_val) & (gray < max_val)

        result[mask] = (gray[mask] - min_val) / (max_val - min_val) * 255

        result = result.astype(np.uint8)

        # 转回三通道
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result

    def glass(self, src):
        h, w = src.shape[:2]
        if h < 7 or w < 7:
            return src.copy()
        glass_img = src.copy()
        rng = np.random.randint(0, 6, size=(h - 6, w - 6), dtype=np.int32)
        ii = np.arange(h - 6, dtype=np.int32)[:, None]
        jj = np.arange(w - 6, dtype=np.int32)[None, :]
        glass_img[: h - 6, : w - 6] = src[ii + rng, jj + rng]
        return glass_img

    def relief(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = gray.astype(np.int16)
        edge = g[:, :-1] - g[:, 1:]
        val = np.clip(edge + 120, 0, 255).astype(np.uint8)
        relief_gray = np.zeros_like(gray, dtype=np.uint8)
        relief_gray[:, :-1] = val
        relief_gray[:, -1] = gray[:, -1]
        return cv2.cvtColor(relief_gray, cv2.COLOR_GRAY2BGR)

    def oil(self, src):
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        h, w = src.shape[:2]
        oil_img = np.zeros_like(src)
        for i in range(2, h - 2):
            for j in range(2, w - 2):
                patch = gray[i - 2 : i + 3, j - 2 : j + 3]
                quant = np.zeros(8, dtype=np.int32)
                for ky in range(5):
                    for kx in range(5):
                        level = int(patch[ky, kx]) // 32
                        level = min(level, 7)
                        quant[level] += 1
                val_index = int(np.argmax(quant))
                lo, hi = val_index * 32, (val_index + 1) * 32
                found = False
                for ky in range(-2, 3):
                    for kx in range(-2, 3):
                        gv = int(gray[i + ky, j + kx])
                        if lo <= gv < hi:
                            oil_img[i, j] = src[i + ky, j + kx]
                            found = True
                            break
                    if found:
                        break
                if not found:
                    oil_img[i, j] = src[i, j]
        oil_img[:2, :] = src[:2, :]
        oil_img[-2:, :] = src[-2:, :]
        oil_img[:, :2] = src[:, :2]
        oil_img[:, -2:] = src[:, -2:]
        return oil_img

    def mosaic(self, src, block=5):
        h, w = src.shape[:2]
        b = max(2, int(block))
        dst = src.copy()
        for i in range(0, h, b):
            for j in range(0, w, b):
                color = src[i, j].copy()
                dst[i : i + b, j : j + b] = color
        return dst

    def sketch(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        temp = 255 - gray
        gauss = cv2.GaussianBlur(temp, (21, 21), 0)
        inver_gauss = 255 - gauss
        sk = cv2.divide(gray, inver_gauss, scale=127.0)
        return cv2.cvtColor(sk, cv2.COLOR_GRAY2BGR)

    def old(self, src):
        b, g, r = cv2.split(src.astype(np.float64))
        new_b = 0.272 * r + 0.534 * g + 0.131 * b
        new_g = 0.349 * r + 0.686 * g + 0.168 * b
        new_r = 0.393 * r + 0.769 * g + 0.189 * b
        merged = cv2.merge([new_b, new_g, new_r])
        return np.clip(merged, 0, 255).astype(np.uint8)

    def lighting(self, img, strength=200):
        rows, cols = img.shape[:2]
        center_x, center_y = rows / 2.0, cols / 2.0
        radius = min(center_x, center_y)
        if radius <= 0:
            return img.copy()
        dst = img.astype(np.float32)
        yy, xx = np.ogrid[:rows, :cols]
        dist = np.sqrt((center_y - xx) ** 2 + (center_x - yy) ** 2)
        mask = dist < radius
        boost = strength * (1.0 - dist[mask] / radius)
        dst[mask] = np.clip(dst[mask] + boost[:, np.newaxis], 0, 255)
        return dst.astype(np.uint8)

    def cartoonize(self, img):
        img_color = img.copy()
        for _ in range(7):
            img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 7)
        edge = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
        )
        edge_color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(img_color, edge_color)