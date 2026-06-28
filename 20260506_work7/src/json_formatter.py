"""JSON格式化模块：将分割结果序列化为COCO兼容格式"""

import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

try:
    from pycocotools import mask as mask_utils
    _HAS_PYCOCOTOOLS = True
except ImportError:
    _HAS_PYCOCOTOOLS = False


# ------------------------------------------------------------------
# JSON 编码器
# ------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """处理 NumPy 类型的 JSON 序列化"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8")
        return super().default(obj)


# ------------------------------------------------------------------
# 掩码编码工具
# ------------------------------------------------------------------

def _encode_rle(mask: np.ndarray) -> dict:
    """
    将二值掩码编码为 COCO RLE 格式。
    优先使用 pycocotools，不可用时回退到纯 Python 实现。
    """
    if _HAS_PYCOCOTOOLS:
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    # 纯 Python 回退：列优先（Fortran 顺序）游程编码
    pixels = mask.T.flatten().astype(np.uint8)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return {
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "counts": runs.tolist(),
    }


def _mask_to_polygons(mask: np.ndarray, epsilon: float = 2.0) -> list:
    """
    使用 OpenCV findContours 将掩码转换为多边形坐标列表。

    Returns:
        [[x1,y1,x2,y2,...], ...] 每个子列表为一个多边形的扁平坐标
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        coords = approx.reshape(-1).tolist()
        if len(coords) >= 6:  # 至少 3 个点
            polygons.append(coords)
    return polygons


def _compute_bbox(mask: np.ndarray) -> list:
    """计算掩码的轴对齐边界框 [x, y, w, h]"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]


# ------------------------------------------------------------------
# 主接口
# ------------------------------------------------------------------

def format_coco_json(
    image_path: str,
    image_shape: tuple,
    masks: np.ndarray,
    scores: np.ndarray,
    category_name: str = "object",
    polygon_epsilon: float = 2.0,
) -> dict:
    """
    将分割结果格式化为 COCO 实例分割 JSON 结构

    Args:
        image_path: 输入图像路径（用于提取文件名）
        image_shape: (height, width)
        masks: (N, H, W) bool 数组
        scores: (N,) 置信度分数
        category_name: 类别名称
        polygon_epsilon: 多边形近似精度（像素）

    Returns:
        dict: 符合 COCO 格式的标注字典，包含：
              info / images / annotations / categories
              annotations 中每条包含：
                id, image_id, category_id,
                segmentation（多边形）, rle（游程编码）,
                bbox, area, score, iscrowd
    """
    h, w = image_shape

    data = {
        "info": {
            "description": "SAM自动分割结果",
            "version": "1.0",
            "date_created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "images": [
            {
                "id": 1,
                "file_name": Path(image_path).name,
                "height": h,
                "width": w,
            }
        ],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": category_name,
                "supercategory": "thing",
            }
        ],
    }

    ann_id = 1
    for mask, score in zip(masks, scores):
        area = int(mask.sum())
        if area == 0:
            continue

        annotation = {
            "id": ann_id,
            "image_id": 1,
            "category_id": 1,
            "segmentation": _mask_to_polygons(mask, epsilon=polygon_epsilon),
            "rle": _encode_rle(mask),
            "bbox": _compute_bbox(mask),
            "area": area,
            "score": float(score),
            "iscrowd": 0,
        }
        data["annotations"].append(annotation)
        ann_id += 1

    return data


def save_json(data: dict, output_path: str) -> str:
    """
    将标注字典序列化为 UTF-8 JSON 文件

    Args:
        data: format_coco_json 返回的字典
        output_path: 输出文件路径

    Returns:
        output_path: 实际写入路径
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=_NumpyEncoder, indent=2, ensure_ascii=False)

    return str(path)
