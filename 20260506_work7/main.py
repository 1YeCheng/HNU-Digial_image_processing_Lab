#!/usr/bin/env python3
"""
SAM图像分割主程序

用法示例：
  # 点提示（默认图像中心）
  python main.py -i inputs/cloud.jpg --model-type vit_h --checkpoint models/sam_vit_h_4b8939.pth

  # 指定 GPU
  python main.py -i inputs/cloud.jpg --device cuda:1 --model-type vit_h --checkpoint models/sam_vit_h_4b8939.pth

  # 指定前景点
  python main.py -i inputs/cloud.jpg --point 500 375

  # 多点（前景+背景）
  python main.py -i inputs/cloud.jpg --point 500 375 --neg-point 100 100

  # 边界框提示
  python main.py -i inputs/cloud.jpg --prompt-mode box --box 100 100 600 500

  # 自动分割（无需提示）
  python main.py -i inputs/cloud.jpg --prompt-mode auto --device cuda:1

  # 保存全部候选掩码
  python main.py -i inputs/cloud.jpg --save-all
"""

import argparse
import sys
import time
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_sam_model
from src.segmentor import SAMSegmentor
from src.mask_exporter import save_masks, save_mask_npy, save_visualization
from src.json_formatter import format_coco_json, save_json


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM图像分割工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", "-i", required=True, help="输入图像路径")
    parser.add_argument(
        "--checkpoint", "-c", default=None,
        help="模型权重路径（默认从配置文件读取）",
    )
    parser.add_argument(
        "--model-type", "-m", default=None,
        choices=["vit_b", "vit_l", "vit_h"],
        help="模型类型（默认从配置文件读取）",
    )
    parser.add_argument(
        "--device", default=None,
        help="推理设备，支持 auto / cpu / cuda / cuda:0 / cuda:1 / cuda:2 / cuda:3（默认 auto）",
    )
    parser.add_argument(
        "--prompt-mode", default="point",
        choices=["point", "box", "auto"],
        help="提示模式（默认: point）",
    )
    parser.add_argument(
        "--point", nargs=2, type=float, metavar=("X", "Y"),
        action="append", dest="points",
        help="前景点坐标，可多次指定，如 --point 500 375",
    )
    parser.add_argument(
        "--neg-point", nargs=2, type=float, metavar=("X", "Y"),
        action="append", dest="neg_points",
        help="背景点坐标，可多次指定",
    )
    parser.add_argument(
        "--box", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
        help="边界框坐标（prompt-mode=box 时使用）",
    )
    parser.add_argument(
        "--output-dir", "-o", default="outputs",
        help="输出根目录（默认: outputs）",
    )
    parser.add_argument(
        "--save-all", action="store_true",
        help="保存全部候选掩码（默认仅保存最优）",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="配置文件路径（默认: configs/default.yaml）",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def resolve_param(cli_val, config_val, default):
    if cli_val is not None:
        return cli_val
    if config_val is not None:
        return config_val
    return default


def run(args, cfg):
    t_start = time.perf_counter()

    model_cfg = cfg.get("model", {})
    out_cfg = cfg.get("output", {})
    vis_cfg = cfg.get("visualization", {})
    json_cfg = cfg.get("json", {})
    infer_cfg = cfg.get("inference", {})

    checkpoint = resolve_param(args.checkpoint, model_cfg.get("checkpoint"), "models/sam_vit_b_01ec64.pth")
    model_type = resolve_param(args.model_type, model_cfg.get("type"), "vit_b")
    device_str = resolve_param(args.device, model_cfg.get("device"), "auto")
    output_dir = args.output_dir
    save_all = args.save_all or out_cfg.get("save_all_masks", False)

    masks_dir = str(Path(output_dir) / "masks")
    vis_dir = str(Path(output_dir) / "vis")
    json_dir = str(Path(output_dir) / "json")

    # ---- 模型加载 ----
    log.info(f"加载模型: {model_type}  权重: {checkpoint}")
    t0 = time.perf_counter()
    predictor, device = load_sam_model(checkpoint, model_type, device_str)
    log.info(f"模型加载完成  设备: {device}  耗时: {time.perf_counter()-t0:.2f}s")

    segmentor = SAMSegmentor(predictor)

    # ---- 图像加载（OpenCV） ----
    log.info(f"读取图像: {args.image}")
    image_rgb, image_bgr = SAMSegmentor.load_image(args.image)
    h, w = image_rgb.shape[:2]
    max_dim = 2048
    if max(h, w) > max_dim:
      scale = max_dim / max(h, w)
      new_h, new_w = int(h * scale), int(w * scale)
      image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
      image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
      h, w = new_h, new_w
      log.info(f"图像缩放至: {new_w}x{new_h}")

    log.info(f"图像尺寸: {w}x{h}")

    basename = Path(args.image).stem

    # ---- 图像嵌入提取 ----
    if args.prompt_mode != "auto":
        log.info("提取图像嵌入...")
        t0 = time.perf_counter()
        segmentor.set_image(image_rgb)
        log.info(f"嵌入提取完成  耗时: {time.perf_counter()-t0:.2f}s")

    # ---- 推理 ----
    log.info(f"执行分割推理  模式: {args.prompt_mode}")
    t0 = time.perf_counter()

    points_arr = None
    labels_arr = None

    if args.prompt_mode == "point":
        all_points = []
        all_labels = []

        if args.points:
            all_points.extend(args.points)
            all_labels.extend([1] * len(args.points))
        else:
            all_points.append([w / 2, h / 2])
            all_labels.append(1)
            log.info(f"未指定提示点，使用图像中心: ({w//2}, {h//2})")

        if args.neg_points:
            all_points.extend(args.neg_points)
            all_labels.extend([0] * len(args.neg_points))

        points_arr = np.array(all_points, dtype=np.float32)
        labels_arr = np.array(all_labels, dtype=np.int32)

        multimask = infer_cfg.get("multimask_output", True)
        masks, scores = segmentor.predict_with_points(all_points, all_labels, multimask_output=multimask)

    elif args.prompt_mode == "box":
        if args.box is None:
            log.error("--prompt-mode box 需要提供 --box X1 Y1 X2 Y2")
            sys.exit(1)
        masks, scores = segmentor.predict_with_box(args.box)

    elif args.prompt_mode == "auto":
        masks, scores, _ = segmentor.predict_auto(image_rgb)

    log.info(f"推理完成  候选掩码: {len(masks)}  耗时: {time.perf_counter()-t0:.2f}s")

    if len(masks) == 0:
        log.warning("未生成任何掩码，请检查提示参数或图像内容")
        sys.exit(0)

    best_idx = int(np.argmax(scores))
    log.info(f"最优掩码索引: {best_idx}  置信度: {scores[best_idx]:.4f}")

    # ---- 掩码输出 ----
    mask_paths = save_masks(masks, scores, masks_dir, basename, save_all=save_all)
    npy_path = save_mask_npy(masks, masks_dir, basename)
    log.info(f"掩码已保存: {mask_paths}")

    # ---- 可视化 ----
    vis_path = save_visualization(
        image_bgr, masks, scores, vis_dir, basename,
        points=points_arr, labels=labels_arr,
        mask_color=tuple(vis_cfg.get("mask_color", [0, 0, 255])),
        mask_alpha=vis_cfg.get("mask_alpha", 0.4),
        contour_color=tuple(vis_cfg.get("contour_color", [0, 255, 0])),
        contour_thickness=vis_cfg.get("contour_thickness", 2),
        point_radius=vis_cfg.get("point_radius", 8),
    )
    log.info(f"可视化已保存: {vis_path}")

    # ---- JSON 输出 ----
    json_data = format_coco_json(
        image_path=args.image,
        image_shape=(h, w),
        masks=masks,
        scores=scores,
        category_name=json_cfg.get("category_name", "object"),
        polygon_epsilon=json_cfg.get("polygon_epsilon", 2.0),
    )
    json_path = save_json(json_data, Path(json_dir) / f"{basename}.json")
    log.info(f"JSON已保存: {json_path}")

    total = time.perf_counter() - t_start
    log.info(
        f"\n{'='*50}\n"
        f"处理完成  总耗时: {total:.2f}s\n"
        f"  掩码:    {mask_paths}\n"
        f"  NumPy:   {npy_path}\n"
        f"  可视化:  {vis_path}\n"
        f"  JSON:    {json_path}\n"
        f"{'='*50}"
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run(args, cfg)


if __name__ == "__main__":
    main()
