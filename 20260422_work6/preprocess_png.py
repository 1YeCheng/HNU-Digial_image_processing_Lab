"""
preprocess_png.py — 针对 PNG 序列的肺部 CT 预处理
将 references/preprocess.py 的 DICOM/MHD+HU 逻辑完整适配为 PNG(0-255) 输入。

主要改动：
  1. load_itk_image  → load_png_series   从文件夹读取 PNG 序列堆叠为 3D 数组
  2. binarize        → 阈值从 HU -600 改为 PNG 像素值 < 90（对应 HU ≈ -600）
  3. resample        → PNG 默认 spacing=[1,1,1]，禁用重采样（do_resample=False）
  4. apply_mask      → pad_value 从 HU 对应的 170 改为 PNG 的 170（软组织均值，保持不变）
  5. generate_label  → 从 validated_nodules.npy（2D 逐切片标注）聚合为 3D bbox
  所有核心形态学逻辑（fill_hole / convex_hull_dilate / seperate_two_lung）保持不变。
"""

import numpy as np
import scipy.ndimage
from skimage import measure, morphology
from scipy.ndimage import label as ndi_label
from pathlib import Path
from PIL import Image
import os
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# 1. 数据加载（替换 load_itk_image）
# ─────────────────────────────────────────────────────────────────────────────

def load_png_series(folder, target_size=512):
    """
    从文件夹读取 PNG 序列，堆叠为 3D uint8 数组 [z, y, x]。

    原 load_itk_image 返回 (image, origin, spacing)，此函数保持相同签名：
      - origin  固定为 [0, 0, 0]（PNG 无物理坐标）
      - spacing 固定为 [1.0, 1.0, 1.0]（像素间距，保持后续 3D 算法兼容）

    参数
    ----
    folder      : str / Path，PNG 切片所在目录
    target_size : int，若切片不是 target_size×target_size 则 resize（默认 512）
    """
    folder = Path(folder)
    files = sorted(folder.glob('*.png'))
    if not files:
        raise FileNotFoundError(f"No PNG files found in {folder}")

    slices = []
    for f in files:
        img = np.array(Image.open(f).convert('L'))   # 强制灰度
        if img.shape[0] != target_size or img.shape[1] != target_size:
            # 可选 resize，避免非 512 分辨率导致后续逻辑出错
            img = cv2.resize(img, (target_size, target_size),
                             interpolation=cv2.INTER_LINEAR)
        slices.append(img)

    volume = np.stack(slices, axis=0).astype(np.uint8)   # [Z, H, W]
    origin  = np.array([0.0, 0.0, 0.0])
    spacing = np.array([1.0, 1.0, 1.0])   # 虚拟 spacing，保持 3D 算法兼容
    return volume, origin, spacing


# ─────────────────────────────────────────────────────────────────────────────
# 2. 二值化（替换 binarize 的 HU 阈值）
# ─────────────────────────────────────────────────────────────────────────────

def binarize(image, spacing,
             intensity_thred=90,    # ← 原 HU -600；PNG 中肺/空气像素值 < 90
             sigma=1.0,
             area_thred=30.0,
             eccen_thred=0.99,
             corner_side=10):
    """
    逐切片二值化 3D PNG 体数据。

    与原版唯一的阈值差异：
      原版：slice_binary = slice_smoothed < -600   （HU，肺+空气为负值）
      本版：slice_binary = slice_smoothed < 90     （PNG，肺+空气为低像素值）

    area_thred 单位为 px²（spacing=[1,1,1] 时 mm² = px²），
    默认 30px² 对应约 6mm 直径，可按需调整。
    """
    binary_mask = np.zeros(image.shape, dtype=bool)
    side_len = image.shape[1]

    grid_axis = np.linspace(-side_len / 2 + 0.5, side_len / 2 - 0.5, side_len)
    x, y = np.meshgrid(grid_axis, grid_axis)
    distance = np.sqrt(np.square(x) + np.square(y))

    # 圆形扫描野之外（四角）置 NaN，防止角落噪声被误判为肺
    nan_mask = (distance < side_len / 2).astype(float)
    nan_mask[nan_mask == 0] = np.nan

    for i in range(image.shape[0]):
        slice_raw = image[i].astype('float32')

        num_uniq = len(np.unique(slice_raw[0:corner_side, 0:corner_side]))
        if num_uniq == 1:
            slice_raw *= nan_mask   # 黑色角落置 NaN

        slice_smoothed = scipy.ndimage.gaussian_filter(slice_raw, sigma, truncate=2.0)

        # ── 核心改动：PNG 低像素值 = 肺/空气（原 HU 负值逻辑保持不变）──
        slice_binary = slice_smoothed < intensity_thred   # 原：< -600

        lbl = measure.label(slice_binary)
        properties = measure.regionprops(lbl)
        label_valid = set()

        for prop in properties:
            # spacing=[1,1,1] 时 area_mm = area_px，单位统一
            area_mm = prop.area * spacing[1] * spacing[2]
            if area_mm > area_thred and prop.eccentricity < eccen_thred:
                label_valid.add(prop.label)

        slice_binary = np.isin(lbl, list(label_valid)).reshape(lbl.shape)
        binary_mask[i] = slice_binary

    return binary_mask


# ─────────────────────────────────────────────────────────────────────────────
# 3. 以下函数与原版完全相同（形态学逻辑不依赖 HU 值）
# ─────────────────────────────────────────────────────────────────────────────

def exclude_corner_middle(label):
    mid = int(label.shape[2] / 2)
    corner_label = set([label[0, 0, 0], label[0, 0, -1],
                        label[0, -1, 0], label[0, -1, -1],
                        label[-1, 0, 0], label[-1, 0, -1],
                        label[-1, -1, 0], label[-1, -1, -1]])
    middle_label = set([label[0, 0, mid], label[0, -1, mid],
                        label[-1, 0, mid], label[-1, -1, mid]])
    for l in corner_label:
        label[label == l] = 0
    for l in middle_label:
        label[label == l] = 0
    return label


def volume_filter(label, spacing, vol_min=0.2, vol_max=8.2):
    """
    按体积过滤连通域。
    spacing=[1,1,1] 时体积单位为 px³，vol_min/vol_max 单位为 1e6 px³（≈ 1L）。
    成人肺体积约 0.55L–3L 每侧，默认范围 [0.2L, 8.2L] 保持不变。
    """
    properties = measure.regionprops(label)
    for prop in properties:
        vol = prop.area * spacing.prod()
        if vol < vol_min * 1e6 or vol > vol_max * 1e6:
            label[label == prop.label] = 0
    return label


def exclude_air(label, spacing, area_thred=3e3, dist_thred=62):
    y_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5,
                         label.shape[1]) * spacing[1]
    x_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5,
                         label.shape[2]) * spacing[2]
    y, x = np.meshgrid(y_axis, x_axis)
    distance = np.sqrt(np.square(y) + np.square(x))
    distance_max = np.max(distance)

    vols = measure.regionprops(label)
    label_valid = set()

    for vol in vols:
        single_vol = (label == vol.label)
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * distance +
                                     (1 - single_vol[i]) * distance_max)
        valid_slices = [min_distance[i] for i in range(label.shape[0])
                        if slice_area[i] > area_thred]
        if valid_slices and np.average(valid_slices) < dist_thred:
            label_valid.add(vol.label)

    binary_mask = np.isin(label, list(label_valid)).reshape(label.shape)
    has_lung = len(label_valid) > 0
    return binary_mask, has_lung


def fill_hole(binary_mask):
    lbl = measure.label(~binary_mask)
    corner_label = set([lbl[0, 0, 0], lbl[0, 0, -1],
                        lbl[0, -1, 0], lbl[0, -1, -1],
                        lbl[-1, 0, 0], lbl[-1, 0, -1],
                        lbl[-1, -1, 0], lbl[-1, -1, -1]])
    binary_mask = ~np.isin(lbl, list(corner_label)).reshape(lbl.shape)
    return binary_mask


def extract_main(binary_mask, cover=0.95):
    for i in range(binary_mask.shape[0]):
        slice_binary = binary_mask[i]
        lbl = measure.label(slice_binary)
        properties = measure.regionprops(lbl)
        properties.sort(key=lambda x: x.area, reverse=True)
        areas = [prop.area for prop in properties]
        count, area_sum = 0, 0
        area_cover = np.sum(areas) * cover
        while area_sum < area_cover:
            area_sum += areas[count]
            count += 1
        slice_filter = np.zeros(slice_binary.shape, dtype=bool)
        for j in range(count):
            r0, c0, r1, c1 = properties[j].bbox
            slice_filter[r0:r1, c0:c1] |= properties[j].convex_image
        binary_mask[i] = binary_mask[i] & slice_filter

    lbl = measure.label(binary_mask)
    properties = measure.regionprops(lbl)
    properties.sort(key=lambda x: x.area, reverse=True)
    binary_mask = (lbl == properties[0].label)
    return binary_mask


def fill_2d_hole(binary_mask):
    for i in range(binary_mask.shape[0]):
        slice_binary = binary_mask[i]
        lbl = measure.label(slice_binary)
        properties = measure.regionprops(lbl)
        for prop in properties:
            r0, c0, r1, c1 = prop.bbox
            slice_binary[r0:r1, c0:c1] |= prop.image_filled
        binary_mask[i] = slice_binary
    return binary_mask


def seperate_two_lung(binary_mask, spacing, max_iter=22, max_ratio=4.8):
    found = False
    iter_count = 0
    binary_mask_full = np.copy(binary_mask)

    while not found and iter_count < max_iter:
        lbl = measure.label(binary_mask, connectivity=2)
        properties = measure.regionprops(lbl)
        properties.sort(key=lambda x: x.area, reverse=True)
        if (len(properties) > 1 and
                properties[0].area / properties[1].area < max_ratio):
            found = True
            eroded1 = (lbl == properties[0].label)
            eroded2 = (lbl == properties[1].label)
        else:
            binary_mask = scipy.ndimage.binary_erosion(binary_mask)
            iter_count += 1

    if found:
        distance1 = scipy.ndimage.distance_transform_edt(~eroded1, sampling=spacing)
        distance2 = scipy.ndimage.distance_transform_edt(~eroded2, sampling=spacing)
        binary_mask1 = binary_mask_full & (distance1 < distance2)
        binary_mask2 = binary_mask_full & (distance1 > distance2)
        binary_mask1 = extract_main(binary_mask1)
        binary_mask2 = extract_main(binary_mask2)
    else:
        binary_mask1 = binary_mask_full
        binary_mask2 = np.zeros(binary_mask.shape, dtype=bool)

    binary_mask1 = fill_2d_hole(binary_mask1)
    binary_mask2 = fill_2d_hole(binary_mask2)
    return binary_mask1, binary_mask2


def convex_hull_dilate(binary_mask, dilate_factor=1.5, iterations=10):
    """
    逐切片取凸包后膨胀，保留贴壁结节。
    此函数操作的是 bool mask，与像素值无关，无需改动。
    """
    binary_mask_dilated = np.array(binary_mask)
    for i in range(binary_mask.shape[0]):
        slice_binary = binary_mask[i]
        if np.sum(slice_binary) > 0:
            slice_convex = morphology.convex_hull_image(slice_binary)
            if np.sum(slice_convex) <= dilate_factor * np.sum(slice_binary):
                binary_mask_dilated[i] = slice_convex

    struct = scipy.ndimage.generate_binary_structure(3, 1)
    binary_mask_dilated = scipy.ndimage.binary_dilation(
        binary_mask_dilated, structure=struct, iterations=iterations)
    return binary_mask_dilated


def apply_mask(image, binary_mask1, binary_mask2,
               pad_value=170,      # PNG 中软组织均值约 170（原 HU 0 → PNG 170，保持不变）
               bone_thred=210,     # PNG 中骨骼阈值约 210（原 HU 400 → PNG 210）
               remove_bone=False):
    """
    将肺 mask 应用到图像，mask 外区域填充 pad_value。
    pad_value=170 对应原版 HU=0（水），在 PNG 尺度下是软组织均值，视觉合理。
    """
    binary_mask = binary_mask1 | binary_mask2
    binary_mask1_dilated = convex_hull_dilate(binary_mask1)
    binary_mask2_dilated = convex_hull_dilate(binary_mask2)
    binary_mask_dilated = binary_mask1_dilated | binary_mask2_dilated
    binary_mask_extra = binary_mask_dilated ^ binary_mask

    image_new = (image * binary_mask_dilated +
                 pad_value * (~binary_mask_dilated)).astype(np.uint8)

    if remove_bone:
        image_new[image_new * binary_mask_extra > bone_thred] = pad_value

    return image_new


# ─────────────────────────────────────────────────────────────────────────────
# 4. 重采样（PNG 默认禁用）
# ─────────────────────────────────────────────────────────────────────────────

def resample(image, spacing, new_spacing=None, order=1):
    """
    PNG 序列 spacing=[1,1,1]，new_spacing 默认也为 [1,1,1]，即不做重采样。
    若需要降采样可传入 new_spacing=[2,1,1] 等。
    """
    if new_spacing is None:
        new_spacing = [1.0, 1.0, 1.0]   # 默认不重采样
    new_spacing = np.array(new_spacing)

    new_shape = np.round(np.array(image.shape) * spacing / new_spacing).astype(int)
    resample_spacing = spacing * np.array(image.shape) / new_shape
    resize_factor = new_shape / np.array(image.shape)

    image_new = scipy.ndimage.zoom(image, resize_factor,
                                   mode='nearest', order=order)
    return image_new, resample_spacing


# ─────────────────────────────────────────────────────────────────────────────
# 5. 完整提取流程
# ─────────────────────────────────────────────────────────────────────────────

def extract_lung(image, spacing):
    """
    从 3D PNG 体数据中提取左右肺 mask。
    流程与原版完全相同，仅 binarize 阈值已适配 PNG。
    """
    binary_mask = binarize(image, spacing)

    lbl = measure.label(binary_mask, connectivity=1)
    lbl = exclude_corner_middle(lbl)
    lbl = volume_filter(lbl, spacing)
    binary_mask, has_lung = exclude_air(lbl, spacing)
    binary_mask = fill_hole(binary_mask)
    binary_mask1, binary_mask2 = seperate_two_lung(binary_mask, spacing)

    return binary_mask1, binary_mask2, has_lung


# ─────────────────────────────────────────────────────────────────────────────
# 6. 结节标签生成（适配 validated_nodules.npy 格式）
# ─────────────────────────────────────────────────────────────────────────────

def generate_label(nodule_npy_path, iou_thred=0.3, min_slices=2):
    """
    从逐切片 2D 结节标注聚合为 3D bbox 列表。

    输入格式（validated_nodules.npy）：
      shape=(Z,)，每个元素是 list of dict：
        {'center': (row, col), 'area': int, 'bbox': (r0, c0, r1, c1)}

    输出：list of dict，每个 3D 结节：
      {'center_zyx': [z, y, x],   # 3D 中心坐标
       'diameter':   float,        # 最大直径（px）
       'z_range':    [z_min, z_max]}

    聚合策略：
      在相邻切片间按 2D IoU > iou_thred 匹配同一结节，
      连续出现 >= min_slices 张才保留（过滤单切片噪声）。
    """
    data = np.load(nodule_npy_path, allow_pickle=True)
    Z = len(data)

    # 将每个切片的结节转为 [z, r0, c0, r1, c1] 列表
    all_boxes = []
    for z, nods in enumerate(data):
        for n in nods:
            r0, c0, r1, c1 = n['bbox']
            all_boxes.append({'z': z, 'r0': r0, 'c0': c0, 'r1': r1, 'c1': c1,
                              'center': n['center'], 'area': n['area']})

    def iou_2d(a, b):
        ir0 = max(a['r0'], b['r0']); ir1 = min(a['r1'], b['r1'])
        ic0 = max(a['c0'], b['c0']); ic1 = min(a['c1'], b['c1'])
        inter = max(0, ir1 - ir0) * max(0, ic1 - ic0)
        if inter == 0:
            return 0.0
        union = a['area'] + b['area'] - inter
        return inter / union if union > 0 else 0.0

    # 贪心聚合：按 z 顺序，将相邻切片 IoU > 阈值的框合并为同一 3D 结节
    nodule_tracks = []   # 每个 track 是一组跨切片的 box

    for box in sorted(all_boxes, key=lambda x: x['z']):
        matched = False
        for track in nodule_tracks:
            last = track[-1]
            if box['z'] - last['z'] <= 2:   # 允许跨 1 张切片的间隔
                if iou_2d(box, last) > iou_thred:
                    track.append(box)
                    matched = True
                    break
        if not matched:
            nodule_tracks.append([box])

    # 过滤短轨迹，计算 3D bbox
    bboxes = []
    for track in nodule_tracks:
        if len(track) < min_slices:
            continue
        zs  = [b['z']  for b in track]
        rs  = [b['r0'] for b in track] + [b['r1'] for b in track]
        cs  = [b['c0'] for b in track] + [b['c1'] for b in track]
        crs = [b['center'][0] for b in track]
        ccs = [b['center'][1] for b in track]

        z_min, z_max = min(zs), max(zs)
        r_min, r_max = min(rs), max(rs)
        c_min, c_max = min(cs), max(cs)

        center_z = np.mean(zs)
        center_y = np.mean(crs)
        center_x = np.mean(ccs)
        diameter  = max(z_max - z_min + 1,
                        r_max - r_min,
                        c_max - c_min)

        bboxes.append({
            'center_zyx': [center_z, center_y, center_x],
            'diameter':   float(diameter),
            'z_range':    [z_min, z_max],
            'n_slices':   len(track),
        })

    return bboxes


# ─────────────────────────────────────────────────────────────────────────────
# 7. 主流程
# ─────────────────────────────────────────────────────────────────────────────

def load_lung_masks(mask_dir):
    """
    加载 lung_extract.py 已生成的肺实质 mask（灰度 PNG，非零区域为肺）。
    返回 [Z, H, W] bool 数组，以及按中线分离的左右肺 mask。

    说明：PNG 数据中体外空气与肺内空气像素值相同（均为 37-90），
    原版 binarize/exclude_air 依赖 HU 负值无法区分两者，
    因此肺实质分割由专用的 lung_extract.py 完成，此处直接加载其结果。
    """
    mask_dir = Path(mask_dir)
    files = sorted(mask_dir.glob('*.png'))
    slices = [np.array(Image.open(f)) > 0 for f in files]
    full_mask = np.stack(slices, axis=0)   # [Z, H, W] bool

    # 按中线分离左右肺（与 lung_extract.py 一致）
    W = full_mask.shape[2]
    mid = W // 2
    mask1 = full_mask.copy(); mask1[:, :, mid:] = False   # 左肺
    mask2 = full_mask.copy(); mask2[:, :, :mid] = False   # 右肺
    return full_mask, mask1, mask2


def preprocess_png(png_dir, mask_dir, nodule_npy, save_dir, do_resample=False):
    """
    完整预处理流程（PNG 版）。

    参数
    ----
    png_dir     : 原始 PNG 切片目录（如 '3055/'）
    mask_dir    : lung_extract.py 生成的肺实质 mask 目录（如 'results/masks/'）
    nodule_npy  : 结节标注文件（validated_nodules.npy）
    save_dir    : 输出目录
    do_resample : 是否重采样（PNG spacing=[1,1,1]，默认 False）
    """
    os.makedirs(save_dir, exist_ok=True)

    print("加载 PNG 序列...")
    img, origin, spacing = load_png_series(png_dir)
    print(f"  体数据 shape: {img.shape}, spacing: {spacing}")

    print("加载肺实质 mask（lung_extract.py 输出）...")
    # 注：PNG 数据体外空气与肺内空气像素值相同，原版 extract_lung 无法区分，
    # 改为加载 lung_extract.py 专用分割结果。
    full_mask, mask1, mask2 = load_lung_masks(mask_dir)
    print(f"  mask shape: {full_mask.shape}, 有肺切片: {full_mask.any(axis=(1,2)).sum()}")

    print("凸包膨胀（保留贴壁结节）...")
    mask1 = convex_hull_dilate(mask1, dilate_factor=1.5, iterations=3)
    mask2 = convex_hull_dilate(mask2, dilate_factor=1.5, iterations=3)

    print("应用 mask...")
    seg_img = apply_mask(img, mask1, mask2)

    if do_resample:
        print("重采样（PNG 通常不需要）...")
        seg_img, spacing = resample(seg_img, spacing)

    print("生成结节标签...")
    bboxes = generate_label(nodule_npy)
    print(f"  3D 结节数: {len(bboxes)}")
    for i, b in enumerate(bboxes[:5]):
        print(f"    [{i}] center={[f'{float(v):.1f}' for v in b['center_zyx']]}, "
              f"diameter={b['diameter']:.1f}px, z={b['z_range']}, slices={b['n_slices']}")

    # 保存
    np.save(os.path.join(save_dir, 'seg_img.npy'),    seg_img)
    np.save(os.path.join(save_dir, 'lung_mask1.npy'), mask1.astype(np.uint8))
    np.save(os.path.join(save_dir, 'lung_mask2.npy'), mask2.astype(np.uint8))
    np.save(os.path.join(save_dir, 'bboxes.npy'),     np.array(bboxes, dtype=object))
    np.save(os.path.join(save_dir, 'spacing.npy'),    spacing)
    print(f"结果保存至: {save_dir}")
    return seg_img, mask1, mask2, bboxes


if __name__ == '__main__':
    preprocess_png(
        png_dir    = '3055',
        mask_dir   = 'results/masks',
        nodule_npy = 'results/validated_nodules.npy',
        save_dir   = 'results/preprocess_out',
        do_resample= False,
    )
