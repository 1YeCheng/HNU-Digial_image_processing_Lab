"""SAM模型加载模块，使用OpenCV进行图像预处理，PyTorch执行推理"""

import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor


def load_sam_model(checkpoint_path: str, model_type: str = "vit_b", device: str = "auto"):
    """
    加载SAM模型

    Args:
        checkpoint_path: 权重文件路径（.pth）
        model_type: 模型类型，可选 vit_b / vit_l / vit_h
        device: 推理设备，支持 auto / cpu / cuda / cuda:0 / cuda:1 / cuda:2 / cuda:3

    Returns:
        predictor: SamPredictor 实例
        device: 实际使用的设备字符串
    """
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"权重文件不存在: {checkpoint}\n"
            f"请下载模型权重放入 models/ 目录。"
        )

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        device = "cuda:0"

    # 校验 cuda:N 设备编号是否合法
    if device.startswith("cuda:"):
        idx = int(device.split(":")[1])
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境无可用 GPU，请改用 --device cpu")
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU {idx} 不存在，当前共有 {torch.cuda.device_count()} 张 GPU（0~{torch.cuda.device_count()-1}）"
            )

    supported = list(sam_model_registry.keys())
    if model_type not in supported:
        raise ValueError(f"不支持的模型类型: {model_type}，可选: {supported}")

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)
    return predictor, device
