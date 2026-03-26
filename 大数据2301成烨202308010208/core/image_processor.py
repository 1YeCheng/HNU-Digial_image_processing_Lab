import cv2
import numpy as np


class ImageProcessor:

    # ===== 基础变换 =====

    def gray(self, img):
        """灰度化"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def binary(self, img):
        """二值化"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def inverse(self, img):
        """反转"""
        return 255 - img

    def gamma(self, img, gamma=0.5):
        """Gamma变换"""
        lut = np.array(
            [np.clip(((i / 255.0) ** gamma) * 255, 0, 255) for i in range(256)],
            dtype=np.uint8
        )
        return cv2.LUT(img, lut)

    def log_transform(self, img):
        """对数变换"""
        img_f = img.astype(np.float32)
        img_f = np.log1p(img_f)
        img_f = cv2.normalize(img_f, None, 0, 255, cv2.NORM_MINMAX)
        return img_f.astype(np.uint8)

    def exp_transform(self, img):
        """指数变换"""
        img_f = img.astype(np.float32) / 255.0
        img_f = np.exp(img_f)
        img_f = cv2.normalize(img_f, None, 0, 255, cv2.NORM_MINMAX)
        return img_f.astype(np.uint8)

    def resize_half(self, img):
        """缩小一半"""
        return cv2.resize(img, None, fx=0.5, fy=0.5)

    def window_level_transform(self, img, window_width, window_level):
        """窗宽窗位变换（模拟CT）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        min_val = window_level - window_width / 2
        max_val = window_level + window_width / 2
        result = np.clip((gray - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # ===== 特效处理 =====

    def glass(self, src):
        """毛玻璃特效（向量化）：偏移范围加大到15，并做双向随机偏移增强磨砂感"""
        h, w = src.shape[:2]
        offset = 15
        if h < offset + 1 or w < offset + 1:
            return src.copy()
        dst = src.copy()
        rh = h - offset
        rw = w - offset
        rng_i = np.random.randint(0, offset, size=(rh, rw), dtype=np.int32)
        rng_j = np.random.randint(0, offset, size=(rh, rw), dtype=np.int32)
        ii = np.arange(rh, dtype=np.int32)[:, None]
        jj = np.arange(rw, dtype=np.int32)[None, :]
        dst[:rh, :rw] = src[ii + rng_i, jj + rng_j]
        return dst

    def relief(self, img):
        """
        浮雕特效：更强的浮雕核（权重扩大）+ Unsharp Mask 锐化 + CLAHE 局部对比度增强，
        立体感更强烈。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        kernel = np.array([[-4, -2,  0],
                           [-2,  1,  2],
                           [ 0,  2,  4]], dtype=np.float32)
        emboss = cv2.filter2D(gray, -1, kernel)
        emboss = np.clip(emboss + 128, 0, 255).astype(np.uint8)
        # Unsharp Mask 进一步锐化边缘
        blur = cv2.GaussianBlur(emboss, (0, 0), 3)
        emboss = cv2.addWeighted(emboss, 1.8, blur, -0.8, 0)
        # CLAHE 局部对比度增强，比全局均衡更有层次感
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        emboss = clahe.apply(emboss)
        return cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)

    def oil(self, img):
        """
        油画特效：多次双边滤波形成大色块笔触，
        叠加边缘轮廓线，提升饱和度，效果强烈。
        """
        result = img.copy()
        for _ in range(6):
            result = cv2.bilateralFilter(result, d=15, sigmaColor=80, sigmaSpace=80)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
        _, edge_bin = cv2.threshold(edge, 20, 255, cv2.THRESH_BINARY)
        edge_inv = cv2.cvtColor(255 - edge_bin, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return cv2.bitwise_and(result, edge_inv)

    def mask(self, src, block=20):
        """马赛克特效（向量化：下采样再上采样）"""
        h, w = src.shape[:2]
        b = max(4, int(block))
        small = cv2.resize(src, (w // b, h // b), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def sketch(self, img):
        """
        素描特效：dodge blend + CLAHE对比度增强 + Canny边缘叠加，
        线条清晰锐利，对比强烈。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        dodge = cv2.divide(gray, 255 - blur, scale=256.0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(dodge)
        edges = cv2.Canny(gray, 30, 100)
        result = cv2.bitwise_and(enhanced, 255 - edges)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def old(self, src):
        """怀旧特效（向量化，float32）"""
        b, g, r = cv2.split(src.astype(np.float32))
        new_b = 0.272 * r + 0.534 * g + 0.131 * b
        new_g = 0.349 * r + 0.686 * g + 0.168 * b
        new_r = 0.393 * r + 0.769 * g + 0.189 * b
        merged = cv2.merge([new_b, new_g, new_r])
        return np.clip(merged, 0, 255).astype(np.uint8)

    def lighting(self, img, strength=200):
        """光照特效（向量化）"""
        rows, cols = img.shape[:2]
        center_x, center_y = rows / 2.0, cols / 2.0
        radius = min(center_x, center_y)
        if radius <= 0:
            return img.copy()
        dst = img.astype(np.float32)
        yy, xx = np.ogrid[:rows, :cols]
        dist = np.sqrt((center_y - xx) ** 2 + (center_x - yy) ** 2)
        in_mask = dist < radius
        boost = (strength * (1.0 - dist[in_mask] / radius))[:, np.newaxis]
        dst[in_mask] = np.clip(dst[in_mask] + boost, 0, 255)
        return dst.astype(np.uint8)

    def hist_equalize(self, img):
        """直方图均衡化（YCrCb空间只均衡亮度，保留色彩不失真）"""
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def cartoonize(self, img):
        """卡通特效"""
        img_color = img.copy()
        for _ in range(3):
            img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 7)
        edge = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
        )
        edge_color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(img_color, edge_color)
