import cv2
import numpy as np
import matplotlib.pyplot as plt


def cv_imread(file_path):
    """支持中文路径读取"""
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def process_full_pipeline(image_path):
    steps = {}

    # 1. Original
    original_img = cv_imread(image_path)
    if original_img is None: return
    steps['1_Original'] = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # 2. Grayscale
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    steps['2_Grayscale'] = gray_img

    # 3. Binary
    _, binary_img = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
    steps['3_Binary'] = binary_img

    # 4. Screen Morph
    screen_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    closed_screen = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, screen_kernel)
    steps['4_Screen_Morph'] = closed_screen

    # 5. Refined Mask (Shrinked to clean edges)
    contours, _ = cv2.findContours(closed_screen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    screen_contour = max(contours, key=cv2.contourArea)
    screen_mask = np.zeros_like(gray_img)
    cv2.drawContours(screen_mask, [screen_contour], -1, 255, thickness=cv2.FILLED)
    screen_mask = cv2.erode(screen_mask, np.ones((15, 15), np.uint8))
    steps['5_Refined_Mask'] = screen_mask

    # 6. Clean Text
    binary_inv = cv2.bitwise_not(binary_img)
    text_cleaned = cv2.bitwise_and(binary_inv, binary_inv, mask=screen_mask)
    steps['6_Clean_Text'] = text_cleaned

    # 7. Block Morph
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 3))
    text_blocks_img = cv2.morphologyEx(text_cleaned, cv2.MORPH_CLOSE, text_kernel)
    steps['7_Block_Morph'] = text_blocks_img

    # 8. Final Extraction
    all_contours, _ = cv2.findContours(text_blocks_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for cnt in all_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 15 and h > 15:  # 适当调低门槛以捕捉底部的单位 't'
            blocks.append({'box': (x, y, w, h), 'cy': y + h // 2})

    # 按Y坐标分组（划分为行）
    blocks.sort(key=lambda b: b['cy'])
    rows = []
    if blocks:
        curr_row = [blocks[0]]
        for i in range(1, len(blocks)):
            if abs(blocks[i]['cy'] - curr_row[-1]['cy']) < 40:  # 行间距阈值
                curr_row.append(blocks[i])
            else:
                rows.append(curr_row)
                curr_row = [blocks[i]]
        rows.append(curr_row)

    # 绘制结果
    final_img = original_img.copy()
    labels = ["Flow", "Temp", "Press"]

    for i, row in enumerate(rows):
        row.sort(key=lambda b: b['box'][0])  # 行内按左到右排序

        # 处理前三行（标签 + 数值）
        if i < 3:
            if len(row) >= 2:
                # 圈出标签 (Label - Red)
                lx, ly, lw, lh = row[0]['box']
                cv2.rectangle(final_img, (lx, ly), (lx + lw, ly + lh), (0, 0, 255), 2)
                cv2.putText(final_img, labels[i], (lx, ly - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 圈出值 (Value - Blue)
                vx, vy, vw, vh = row[1]['box']
                cv2.rectangle(final_img, (vx, vy), (vx + vw, vy + vh), (255, 0, 0), 2)
                cv2.putText(final_img, "Value", (vx, vy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 处理第四行及以后（长数字总数行）
        else:
            # 计算这一行所有块的最小包含矩形
            all_x = [b['box'][0] for b in row]
            all_y = [b['box'][1] for b in row]
            all_xw = [b['box'][0] + b['box'][2] for b in row]
            all_yh = [b['box'][1] + b['box'][3] for b in row]

            x_min, y_min = min(all_x), min(all_y)
            x_max, y_max = max(all_xw), max(all_yh)

            # 圈出整行 (Cyan color)
            cv2.rectangle(final_img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

    steps['8_Final_Result'] = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

    # 显示结果
    plt.figure(figsize=(20, 10))
    for i, (title, img) in enumerate(steps.items()):
        plt.subplot(2, 4, i + 1)
        plt.title(title)
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    img_path = "D:/hnu/third/数据图像处理/work/20260414_work5/screen_image.jpg"  # 替换为你的路径
    process_full_pipeline(img_path)