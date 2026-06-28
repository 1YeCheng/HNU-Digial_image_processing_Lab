"""掩码导出模块：PNG保存、NumPy存储、可视化叠加"""

from pathlib import Path
import cv2
import numpy as np


def save_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    output_dir: str,
    basename: str,
    save_all: bool = False,
):
    """
    将分割掩码保存为 PNG 文件

    Args:
        masks: (N, H, W) bool 数组
        scores: (N,) 置信度
        output_dir: 输出目录
        basename: 文件名前缀（不含扩展名）
        save_all: True=保存全部候选，False=仅保存最优

    Returns:
        saved_paths: 已保存文件路径列表
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    if len(masks) == 0:
        return saved_paths

    indices = range(len(masks)) if save_all else [int(np.argmax(scores))]

    for i in indices:
        mask_uint8 = (masks[i].astype(np.uint8)) * 255
        suffix = f"_mask_{i}" if save_all else "_mask"
        path = out / f"{basename}{suffix}.png"
        cv2.imwrite(str(path), mask_uint8)
        saved_paths.append(str(path))

    return saved_paths


def save_mask_npy(masks: np.ndarray, output_dir: str, basename: str):
    """保存掩码为 NumPy 二进制格式（保留完整精度）"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{basename}_masks.npy"
    np.save(str(path), masks)
    return str(path)


def save_visualization(
    image_bgr: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    output_dir: str,
    basename: str,
    points: np.ndarray = None,
    labels: np.ndarray = None,
    mask_color: tuple = (0, 0, 255),
    mask_alpha: float = 0.4,
    contour_color: tuple = (0, 255, 0),
    contour_thickness: int = 2,
    point_radius: int = 8,
):
    """
    生成原图 + 掩码半透明叠加的可视化图像

    Args:
        image_bgr: 原始 BGR 图像
        masks: (N, H, W) bool 数组
        scores: (N,) 置信度
        output_dir: 输出目录
        basename: 文件名前缀
        points: [[x,y],...] 提示点坐标（可选）
        labels: [1/0,...] 提示点标签（可选）

    Returns:
        vis_path: 可视化图像路径
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if len(masks) == 0:
        path = out / f"{basename}_vis.jpg"
        cv2.imwrite(str(path), image_bgr)
        return str(path)

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]

    vis = image_bgr.copy()

    # 半透明掩码叠加
    color_layer = np.zeros_like(vis)
    color_layer[best_mask] = mask_color
    vis = cv2.addWeighted(vis, 1.0 - mask_alpha, color_layer, mask_alpha, 0)

    # 轮廓描边（使用 OpenCV findContours）
    mask_uint8 = best_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, contour_color, contour_thickness)

    # 绘制提示点
    if points is not None and labels is not None:
        for pt, lbl in zip(points, labels):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(vis, (cx, cy), point_radius, color, -1)
            cv2.circle(vis, (cx, cy), point_radius, (255, 255, 255), 2)

    # 置信度文字
    score_text = f"score: {scores[best_idx]:.3f}"
    cv2.putText(vis, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    path = out / f"{basename}_vis.jpg"
    cv2.imwrite(str(path), vis)
    return str(path)
