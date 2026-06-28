"""SAM推理引擎，封装点提示、框提示和自动分割三种模式"""

import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator


class SAMSegmentor:
    """
    SAM分割推理器

    使用 OpenCV 完成图像 I/O 和预处理（BGR→RGB、格式校验），
    通过 SamPredictor 执行模型推理。
    """

    def __init__(self, predictor):
        self.predictor = predictor
        self.sam = predictor.model
        self._image_set = False
        self.image_shape = None  # (H, W)

    # ------------------------------------------------------------------
    # 图像设置
    # ------------------------------------------------------------------

    def set_image(self, image_rgb: np.ndarray):
        """
        设置输入图像并提取嵌入（耗时操作，结果会被缓存）

        Args:
            image_rgb: uint8 RGB 图像，形状 (H, W, 3)
        """
        if image_rgb.dtype != np.uint8:
            raise TypeError(f"图像数据类型须为 uint8，当前: {image_rgb.dtype}")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"图像须为 3 通道，当前形状: {image_rgb.shape}")

        self.predictor.set_image(image_rgb)
        self._image_set = True
        self.image_shape = image_rgb.shape[:2]

    @staticmethod
    def load_image(image_path: str):
        """
        使用 OpenCV 加载图像并转换为 RGB

        Args:
            image_path: 图像文件路径（支持中文路径）

        Returns:
            image_rgb: uint8 RGB 图像
            image_bgr: uint8 BGR 图像（用于可视化保存）
        """
        # 使用 np.fromfile + imdecode 支持中文路径
        img_array = np.fromfile(image_path, dtype=np.uint8)
        image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        # 处理 RGBA（4通道）图像
        if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb, image_bgr

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    def predict_with_points(
        self,
        points: list,
        labels: list,
        multimask_output: bool = True,
    ):
        """
        点提示分割

        Args:
            points: [[x1,y1], [x2,y2], ...] 像素坐标列表
            labels: [1, 0, ...] 1=前景，0=背景
            multimask_output: True 时返回 3 个候选掩码

        Returns:
            masks: (N, H, W) bool 数组
            scores: (N,) 置信度
        """
        self._check_image_set()
        pts = np.array(points, dtype=np.float32)
        lbs = np.array(labels, dtype=np.int32)
        self._validate_points(pts)

        masks, scores, _ = self.predictor.predict(
            point_coords=pts,
            point_labels=lbs,
            multimask_output=multimask_output,
        )
        return masks, scores

    def predict_with_box(self, box: list, multimask_output: bool = False):
        """
        边界框提示分割

        Args:
            box: [x1, y1, x2, y2] 像素坐标
            multimask_output: 是否输出多候选

        Returns:
            masks: (N, H, W) bool 数组
            scores: (N,) 置信度
        """
        self._check_image_set()
        box_arr = np.array(box, dtype=np.float32)[None, :]  # (1, 4)

        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_arr,
            multimask_output=multimask_output,
        )
        return masks, scores

    def predict_auto(self, image_rgb: np.ndarray, **kwargs):
        """
        自动分割模式（无需提示，对全图生成所有可能的掩码）

        Args:
            image_rgb: uint8 RGB 图像
            **kwargs: 传递给 SamAutomaticMaskGenerator 的参数

        Returns:
            masks: (N, H, W) bool 数组
            scores: (N,) stability_score
            raw_results: 原始结果列表（含 bbox、area 等）
        """
        generator = SamAutomaticMaskGenerator(self.sam, **kwargs)
        raw_results = generator.generate(image_rgb)

        if not raw_results:
            h, w = image_rgb.shape[:2]
            return np.zeros((0, h, w), dtype=bool), np.array([]), []

        masks = np.array([r["segmentation"] for r in raw_results])
        scores = np.array([r["stability_score"] for r in raw_results])
        return masks, scores, raw_results

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _check_image_set(self):
        if not self._image_set:
            raise RuntimeError("请先调用 set_image() 设置输入图像")

    def _validate_points(self, points: np.ndarray):
        if self.image_shape is None:
            return
        h, w = self.image_shape
        if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= w):
            raise ValueError(f"点的 x 坐标超出图像宽度范围 [0, {w})")
        if np.any(points[:, 1] < 0) or np.any(points[:, 1] >= h):
            raise ValueError(f"点的 y 坐标超出图像高度范围 [0, {h})")
