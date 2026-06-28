# -*- coding: utf-8 -*-
import cv2
import numpy as np


class ImageProcessor:

    # ── 傅里叶频谱 ────────────────────────────────────────────────────────────

    def compute_spectrum(self, img: np.ndarray) -> np.ndarray:
        """计算图像的傅里叶频谱（对数幅度，MAGMA 伪彩色，固定 512×512 输出）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        dft_shift = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        log_mag = np.log1p(np.abs(dft_shift))
        normalized = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)
        return cv2.resize(colored, (512, 512), interpolation=cv2.INTER_LINEAR)

    # ── 空间域变换 ────────────────────────────────────────────────────────────

    def zoom(self, img: np.ndarray, factor: float) -> np.ndarray:
        """物理缩放"""
        h, w = img.shape[:2]
        new_w, new_h = max(5, int(w * factor)), max(5, int(h * factor))
        interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    def translate(self, img: np.ndarray, tx: int, ty: int) -> np.ndarray:
        """平移（扩充画布）"""
        h, w = img.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(img, M, (w + abs(tx), h + abs(ty)), borderValue=(255, 255, 255))

    def rotate(self, img: np.ndarray, angle: float) -> np.ndarray:
        """旋转（自动扩充画布防裁剪）"""
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        nW, nH = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += nW / 2 - w / 2
        M[1, 2] += nH / 2 - h / 2
        return cv2.warpAffine(img, M, (nW, nH), borderValue=(255, 255, 255))

    def gaussian_smooth(self, img: np.ndarray, ksize: int = 15, sigma: float = 0) -> np.ndarray:
        """高斯平滑"""
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    def laplacian_sharpen(self, img: np.ndarray) -> np.ndarray:
        """拉普拉斯锐化（保留原图 + 增强边缘）"""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    def median_denoise(self, img: np.ndarray, ksize: int = 5) -> np.ndarray:
        """中值去噪"""
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.medianBlur(img, ksize)

    # ── 频率域滤波 ────────────────────────────────────────────────────────────

    def _freq_filter(self, img: np.ndarray, H: np.ndarray) -> np.ndarray:
        """通用频率域滤波：gray → FFT → ×H → IFFT → BGR"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        dft_shift = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        filtered = np.fft.ifft2(np.fft.ifftshift(dft_shift * H))
        result = np.clip(np.abs(filtered), 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def _butterworth_mask(self, shape, cutoff: int, order: int, lowpass: bool) -> np.ndarray:
        """构建 Butterworth 滤波器掩模"""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        V, U = np.meshgrid(v, u)
        D = np.sqrt(U ** 2 + V ** 2)
        if lowpass:
            H = 1.0 / (1.0 + (D / cutoff) ** (2 * order))
        else:
            D = np.maximum(D, 1e-6)
            H = 1.0 / (1.0 + (cutoff / D) ** (2 * order))
        return H

    def butterworth_lowpass(self, img: np.ndarray, cutoff: int = 30, order: int = 2) -> np.ndarray:
        """Butterworth 低通滤波"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        H = self._butterworth_mask(gray.shape, cutoff, order, lowpass=True)
        return self._freq_filter(img, H)

    def butterworth_highpass(self, img: np.ndarray, cutoff: int = 30, order: int = 2) -> np.ndarray:
        """Butterworth 高通滤波"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        H = self._butterworth_mask(gray.shape, cutoff, order, lowpass=False)
        return self._freq_filter(img, H)
