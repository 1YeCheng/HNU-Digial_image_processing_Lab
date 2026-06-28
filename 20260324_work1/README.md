# 图像处理系统 — Work 1

## 项目概述

基于 PyQt5 的图像处理桌面应用，采用 MVC 分层架构，支持 27 种图像处理操作，并集成 AI 图像评价功能。

---

## 项目结构

```
20260324_work1/
├── main.py                      # 应用入口，启动 PyQt5 主窗口
├── controller/
│   └── image_controller.py      # 控制层，转发 UI 操作到业务层
├── service/
│   └── image_service.py         # 业务逻辑层，管理图像状态与处理流程
├── core/
│   └── image_processor.py       # 核心算法层，实现所有图像处理算法
└── ui/
    └── main_window.py           # UI 层，主窗口与交互逻辑
```

---

## 架构设计

```
UI 层 (main_window.py)
    ↓  用户操作
控制层 (image_controller.py)
    ↓  调用
业务逻辑层 (image_service.py)
    ↓  调用
核心算法层 (image_processor.py)
```

- 控制层负责解耦 UI 与业务，UI 只与 Controller 交互
- 业务层维护两份图像状态：`image`（原始，始终不变）和 `result`（当前处理结果）
- 基础变换和空间变换在 `result` 上累积叠加；特效处理每次基于原始 `image` 独立计算

---

## 功能列表

### 基础变换（累积叠加）

| 功能 | 方法 | 说明 |
|------|------|------|
| 灰度化 | `gray()` | RGB 转灰度 |
| 二值化 | `binary()` | 阈值 128 |
| 反转 | `inverse()` | 255 − pixel |
| Gamma 变换 | `gamma()` | γ = 0.5，增强暗部 |
| 对数变换 | `log_transform()` | 压缩高亮，扩展暗部 |
| 指数变换 | `exp_transform()` | 扩展高亮 |
| 缩小一半 | `resize_half()` | 双线性插值 |
| 直方图均衡化 | `hist_equalize()` | YCrCb 空间只均衡亮度通道 |

### 图像特效（基于原始图像）

| 功能 | 方法 | 核心技术 |
|------|------|---------|
| 毛玻璃 | `glass()` | 随机像素偏移（offset=15px） |
| 浮雕 | `relief()` | 浮雕卷积核 + Unsharp Mask + CLAHE |
| 油画 | `oil()` | 双边滤波 + 边缘轮廓 + 饱和度增强 |
| 马赛克 | `mask()` | 下采样再上采样（block=20px） |
| 素描 | `sketch()` | Dodge Blend + CLAHE + Canny 边缘 |
| 怀旧 | `old()` | 色彩矩阵变换 |
| 光照 | `lighting()` | 径向光照增强（strength=200） |
| 卡通 | `cartoonize()` | 双边滤波 + 自适应阈值 |

### 空间变换（累积叠加）

| 功能 | 方法 | 说明 |
|------|------|------|
| 缩放 | `zoom(factor)` | 任意比例缩放 |
| 旋转 | `rotate(angle)` | 自动扩充画布防裁剪 |
| 平移 | `translate(tx, ty)` | 扩充画布保留完整图像 |
| 水平镜像 | `flip_h()` | — |
| 垂直镜像 | `flip_v()` | — |
| 剪切 | `shear(factor)` | 仿射剪切变换 |
| 透视 | `perspective()` | 梯形畸变 |
| 波浪 | `wave()` | remap 非线性变换 |
| 拼接 | `stitch(img2)` | 特征点匹配拼接，失败时并排合并 |

---

## 运行方式

```bash
pip install PyQt5 opencv-python numpy pillow openai
python main.py
```

---

## UI 说明

- 顶部栏：打开 / 保存 / 恢复原图 / 图像信息
- AI 评价栏：调用大模型对当前处理结果进行图像质量评价
- 图像展示区：原始图像与处理结果并排显示
- 功能卡片区：三组可滚动卡片（基础变换 / 图像特效 / 空间变换），支持悬停效果，处理中自动禁用防止重复点击
- 所有处理操作异步执行，UI 不卡顿
