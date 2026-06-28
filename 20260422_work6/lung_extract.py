"""
肺实质提取 — Patient 3055
输出：背景置0，肺实质区域保留原始CT像素值的灰度图

算法整合自：
  - references/preprocess.py  : convex_hull_dilate / apply_mask / fill_2d_hole / extract_main
  - references/segment_lung.py: 形态学阈值分割 + 连通域选取思路

体外空气隔离策略（参考 preprocess.py exclude_corner_middle 思路）：
  对全图空气做连通域，排除碰到图像边界的连通域（体外空气），
  排除中心行 > H*0.6 的连通域（腹部气体）。

肺实质完整性策略（参考 preprocess.py extract_main 思路）：
  合并同侧所有有效内部空气连通域（而非只取最大），
  再做 binary_fill_holes 填充完整肺轮廓（包含血管、支气管、肺壁）。

贴壁结节保留策略（参考 preprocess.py convex_hull_dilate，iterations=10）：
  1. binary_fill_holes 填充完整肺轮廓
  2. convex_hull_image（精确凸包，面积增幅≤1.5倍时才替换）消除肺边缘大凹陷
  3. 膨胀（r=5px）确保贴壁结节被包含在mask内
  4. 排除骨骼（>200px）避免膨胀后包含肋骨

3D 连续性保障：
  对分割失败（肺和体外空气完全连通）的切片，用相邻切片的 mask 填补。
"""

import numpy as np
from PIL import Image
import scipy.ndimage as ndi
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, binary_erosion
from skimage.morphology import convex_hull_image
from skimage import measure
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────────────────────────────
SLICE_DIR = Path('3055')
OUT_DIR   = Path('results/masks')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 参数 ──────────────────────────────────────────────────────────────────────
AIR_LOW      = 37    # 肺内空气下界
AIR_HIGH     = 90    # 肺内空气上界
MIN_LUNG_PX  = 100   # 肺连通域最小像素数（降低以合并小连通域）
CLOSE_R      = 3     # 闭运算半径（连接相邻空气区域）
DILATE_R     = 5     # 凸包后膨胀半径（参考代码iterations=10，此处2D等效5px）
ERODE_ITER   = 1     # 腐蚀迭代次数（1次足够断开气管/食管与肺的连接，过多会使肺碎片化）
HULL_FACTOR  = 1.5   # 凸包面积增幅上限（超过则不替换，防止两肺合并）
ROW_RATIO    = 0.6   # 肺中心行上限（排除腹部气体）


def disk_se(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x**2 + y**2 <= r**2)


def segment_lung_slice(img: np.ndarray):
    """
    对单张CT切片提取左/右肺 bool mask。

    流程：
      1. 全图空气连通域，排除体外空气（碰边界）和腹部气体（中心行>H*0.6）
      2. 合并所有有效内部空气连通域
      3. 腐蚀（ERODE_ITER次）断开气管与肺之间的细支气管连接
      4. 保留最大的两个连通域（左右肺），用距离变换把原始空气像素分配回去
      5. 对每侧：闭运算 → fill_holes → 凸包 → 膨胀 → 排除骨骼
    """
    H, W = img.shape
    empty = np.zeros((H, W), dtype=bool)
    mid = W // 2

    # ── Step 1-2: 收集所有有效内部空气连通域 ─────────────────────────────────
    air = (img >= AIR_LOW) & (img < AIR_HIGH)
    lbl, n = ndi.label(air)
    if n == 0:
        return empty.copy(), empty.copy()

    sz = np.array(ndi.sum(air, lbl, range(1, n + 1)))
    valid_air = empty.copy()
    for l in range(1, n + 1):
        if sz[l - 1] < MIN_LUNG_PX:
            continue
        m = lbl == l
        if m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any():
            continue
        cr = float(np.where(m)[0].mean())
        if cr > H * ROW_RATIO:
            continue
        valid_air |= m

    if valid_air.sum() == 0:
        return empty.copy(), empty.copy()

    # ── Step 3: 腐蚀断开气管与肺之间的细支气管 ──────────────────────────────
    eroded = valid_air.copy()
    for _ in range(ERODE_ITER):
        eroded = binary_erosion(eroded)
        if eroded.sum() == 0:
            break

    # ── Step 4: 保留最大两个连通域，用距离变换分配原始像素 ───────────────────
    lbl_e, n_e = ndi.label(eroded)

    if n_e < 2:
        # 腐蚀后只剩0或1个连通域，按中线直接分割
        left_air  = valid_air.copy(); left_air[:, mid:]  = False
        right_air = valid_air.copy(); right_air[:, :mid] = False
    else:
        sz_e = np.array(ndi.sum(eroded, lbl_e, range(1, n_e + 1)))
        top2 = np.argsort(sz_e)[-2:]          # 最大两个的0-indexed
        comp1 = (lbl_e == (top2[0] + 1))
        comp2 = (lbl_e == (top2[1] + 1))

        # 距离变换：把 valid_air 中每个像素分配给最近的肺连通域
        dist1 = ndi.distance_transform_edt(~comp1)
        dist2 = ndi.distance_transform_edt(~comp2)
        assigned1 = valid_air & (dist1 <= dist2)
        assigned2 = valid_air & (dist1 >  dist2)

        # 按质心列坐标确定左右
        c1 = float(np.where(assigned1)[1].mean()) if assigned1.sum() > 0 else mid
        c2 = float(np.where(assigned2)[1].mean()) if assigned2.sum() > 0 else mid
        if c1 <= c2:
            left_air, right_air = assigned1, assigned2
        else:
            left_air, right_air = assigned2, assigned1

    # ── Step 5: 对每侧做形态学处理 ───────────────────────────────────────────
    se_close   = disk_se(CLOSE_R)
    left_full  = empty.copy()
    right_full = empty.copy()

    for air_side, out in [(left_air, left_full), (right_air, right_full)]:
        if air_side.sum() == 0:
            continue

        # fill_holes 之前先排除孤立小连通域（气管、食管空气等）
        # 它们在距离变换分配后仍是独立连通域，与主肺不相连
        lbl_a, n_a = ndi.label(air_side)
        if n_a > 1:
            sz_a = np.array(ndi.sum(air_side, lbl_a, range(1, n_a + 1)))
            air_side = (lbl_a == (np.argmax(sz_a) + 1))

        m = binary_closing(air_side, structure=se_close)
        m = binary_fill_holes(m)

        # 保留最大连通域（额外保险）
        lbl2, n2 = ndi.label(m)
        if n2 > 1:
            sz2 = np.array(ndi.sum(m, lbl2, range(1, n2 + 1)))
            m = (lbl2 == (np.argmax(sz2) + 1))

        if m.sum() > 0:
            hull = convex_hull_image(m)
            if hull.sum() <= HULL_FACTOR * m.sum():
                m = hull

        m = binary_dilation(m, structure=disk_se(DILATE_R))
        m = m & (img < 200)
        out[:] = m

    return left_full, right_full


def apply_mask(img: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
    """参考 preprocess.py apply_mask()：背景置0，肺实质区域保留原始CT像素值。"""
    result = np.zeros_like(img, dtype=np.uint8)
    result[lung_mask] = img[lung_mask]
    return result


def process_all():
    files = sorted(SLICE_DIR.glob('*.png'))
    n = len(files)
    print(f"共 {n} 张切片，开始提取肺实质...")

    imgs  = [np.array(Image.open(f)) for f in files]
    masks = []

    # ── Pass 1: 逐切片分割 ────────────────────────────────────────────────────
    for i, img in enumerate(imgs):
        left, right = segment_lung_slice(img)
        masks.append(left | right)

    # ── Pass 2: 3D传播，填补失败/异常小切片（多轮直到收敛）────────────────────
    for _ in range(3):
        areas = np.array([m.sum() for m in masks])
        changed = False
        for i in range(n):
            win = areas[max(0, i-5):min(n, i+6)]
            win_med = float(np.median(win[win > 0])) if (win > 0).any() else 0
            if areas[i] > 0 and areas[i] >= win_med * 0.3:
                continue
            prev = next((j for j in range(i - 1, -1, -1) if areas[j] >= win_med * 0.3), None)
            nxt  = next((j for j in range(i + 1, n)      if areas[j] >= win_med * 0.3), None)
            if prev is not None and nxt is not None:
                masks[i] = masks[prev] | masks[nxt]
            elif prev is not None:
                masks[i] = masks[prev].copy()
            elif nxt is not None:
                masks[i] = masks[nxt].copy()
            areas[i] = masks[i].sum()
            changed = True
        if not changed:
            break

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    lung_areas = []
    for i, (fpath, img, lung_mask) in enumerate(zip(files, imgs, masks)):
        result = apply_mask(img, lung_mask)
        Image.fromarray(result).save(OUT_DIR / fpath.name)
        lung_areas.append(int(lung_mask.sum()))

        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] {fpath.name}  lung={lung_mask.sum()}px")

    print(f"\n完成！平均肺面积: {np.mean(lung_areas):.0f} px")
    print(f"结果保存在: {OUT_DIR}")
    return lung_areas


if __name__ == '__main__':
    process_all()
