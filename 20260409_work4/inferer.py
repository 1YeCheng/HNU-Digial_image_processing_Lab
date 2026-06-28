# 这是一个用来进行文字识别的类
import cv2
from template import TemplateMatcher
from model import *
from ConfigManager import ConfigManager
from parser_ocr_no147 import Parser
import numpy as np

class Infer:
    """
    推理类，用来从图像中提取仪表屏幕部分， 检测屏幕中的文字块，然后识别文字，得到OCRCharResults
    """
    def __init__(self, config):
        self.config = config
        self.ocr_reg = TemplateMatcher(
            template_dir=self.config.get_config().get("OCR").get("template_dir"),
            img_size=(40, 80),
            method=self.config.get_config().get("OCR").get("ocr_method"),
            threshold=self.config.get_config().get("OCR").get("threshold")
        )

    def label_to_content(self, label):
        """
        将 label 转换为对应 content
        如果 label 不在 mapping 中，则返回原 label
        """
        label_map = self.config.get_config().get("label_to_chinese")
        return label_map.get(label, label)

    def pre_crop(self, img):
        """
        用来截取一部分图像
        :param img:
        :return:
        """
        # 得到原始图像的尺寸
        k = self.config.get_params("OCR", ["img_w", "img_h", "crop_x0", "crop_y0", "crop_x1", "crop_y1"])
        img = cv2.resize(img, (k.get("img_w"), k.get("img_h")))
        x0 = k.get("crop_x0", 0)
        y0 = k.get("crop_y0", 0)
        x1 = k.get("crop_x1")
        y1 = k.get("crop_y1")

        h, w = img.shape[:2]

        # 防止越界
        if x1 > w or y1 > h:
            return False, img

        # 裁剪
        cropped = img[y0:y1, x0:x1]

        if cropped.size == 0:
            return False, img

        return True, cropped

    def extract_screen(self, img):
        """
        提取img中的屏幕部分
        :param img: 需要推理的图像
        :return:
        """

        # 预处理：转为灰度并模糊处理以减少噪点

        if img is not None:
            print(f"Image shape: {img.shape if hasattr(img, 'shape') else 'No shape attribute'}")
            print(f"Image dtype: {img.dtype if hasattr(img, 'dtype') else 'No dtype'}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 边缘检测或阈值处理
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 寻找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找到面积最大的轮廓（通常就是屏幕）
        if not contours:
            return False, "轮廓检测失败"

        screen_contour = max(contours, key=cv2.contourArea)

        #  获取外接矩形并裁剪
        x, y, w, h = cv2.boundingRect(screen_contour)

        # 为了保险，稍微向内缩进几个像素，避开边缘黑框
        k = self.config.get_params("OCR", ["crop_padding"])
        crop_padding = k.get("crop_padding", 5)
        padding = crop_padding
        cropped = img[y + padding:y + h - padding, x + padding:x + w - padding]
        cropped = cv2.resize(cropped, (820, 456))

        return True, cropped

    def to_binary(self, img):
        """
        :param img:
        :return:
        """
        gray = img.copy()
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _remove_inner_contours(self, contours):
        """
        删除完全被其他轮廓包住的小轮廓
        参数:
            contours: findContours 返回的轮廓列表
        返回:
            过滤后的轮廓列表
        """

        if len(contours) <= 1:
            return contours

        # 计算每个轮廓的 bounding box
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

        keep = [True] * len(contours)

        for i in range(len(boxes)):
            xi, yi, wi, hi = boxes[i]

            for j in range(len(boxes)):
                if i == j:
                    continue

                xj, yj, wj, hj = boxes[j]

                # 判断 i 是否完全在 j 内部
                if (xi >= xj and
                        yi >= yj and
                        xi + wi <= xj + wj and
                        yi + hi <= yj + hj):

                    # 如果面积更小，则删除 i
                    area_i = wi * hi
                    area_j = wj * hj

                    if area_i < area_j:
                        keep[i] = False
                        break

        filtered_contours = [
            contours[i] for i in range(len(contours)) if keep[i]
        ]

        return filtered_contours

    def text_dect(self, img):
        """
        提取图像中的文本块
        :param img: 图像
        :return:
        """
        gray = img.copy()
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        # 二值化（反色更稳定）
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 小闭运算修复笔画
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        # 横向聚合（核心步骤）
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 9))
        k = self.config.get_params("OCR", ["struct_w", "struct_h"])
        struct_w = k.get("struct_w", 11)
        struct_h = k.get("struct_h", 5)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (struct_w, struct_h))
        blocks = cv2.dilate(binary, kernel, iterations=1)

        # 轮廓检测
        contours, _ = cv2.findContours(
            blocks,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contours = self._remove_inner_contours(contours)  # 过滤大轮廓里的小轮廓

        text_blocks = []

        min_ratio = self.config.get_config().get("OCR").get("min_ratio")
        max_ratio = self.config.get_config().get("OCR").get("max_ratio")
        min_area = self.config.get_config().get("OCR").get("min_area")

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < min_area:
                continue
            ratio = w / h
            if ratio < min_ratio or ratio > max_ratio:
                continue

            text_blocks.append((x, y, w, h))

        text_blocks = sorted(text_blocks, key=lambda b: b[0])

        return text_blocks, gray

    def draw_textblocks(self, text_blocks, screen_img):
        """
        在图像上绘制文本块框

        :param text_blocks: [(x,y,w,h), ...]
        :param screen_img: 图像
        :return: 绘制后的图像
        """

        img = screen_img.copy()

        # 如果是二值图，转成BGR方便画彩色框
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i, (x, y, w, h) in enumerate(text_blocks):
            # 画框
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # 可选：标号
            cv2.putText(
                img,
                str(i),
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        return img

    # 对外接口
    def infer(self, img):
        """

        :param img:
        :return:
        """

        OCR_char_results = []

        k = self.config.get_params("OCR", ["is_crop"])
        is_crop = k.get("is_crop", False)
        if is_crop:
            r, img1 = self.pre_crop(img)

        success, screen_img = self.extract_screen(img1)
        if not success:
            return []

        text_blocks, crop_img = self.text_dect(screen_img)
        screen_img = self.to_binary(screen_img)

        # 增加一个测试函数来查看text_blocks的效果
        box_img = self.draw_textblocks(text_blocks, crop_img)

        for bx, by, bw, bh in text_blocks:
            block_img_bin = screen_img[by:by + bh, bx:bx + bw]
            label, score = self.ocr_reg.predict(block_img_bin)
            label = self.label_to_content(label)
            if label != "unknown":
                OCR_char_results.append(
                    OCRCharResult(bx, by, bw, bh, label, score)
                )
        # box_img = self.draw_ocrs(OCR_char_results, crop_img)

        return OCR_char_results, box_img

    def draw_ocrs(self, ocr_results, screen_img):
        """
        在图像上绘制文本块框，并显示OCR识别结果

        :param ocr_results: [OCRCharResult,...]
        :param screen_img: 图像
        :return: 绘制后的图像
        """

        img = screen_img.copy()

        # 如果是二值图，转成BGR方便画彩色框
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i, res in enumerate(ocr_results):
            x, y, w, h = res.x, res.y, res.w, res.h
            label = res.content
            score = res.confidence

            # print("(x, y, w, h),label,score", x, y, w, h, label, score)

            # 画框
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # 要显示的文字
            text = f"{label}:{score:.2f}"

            # 文字位置
            text_y = y - 5 if y - 5 > 10 else y + 15

            # 绘制文字
            cv2.putText(
                img,
                text,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        return img

#
#
# def main():
#
#     # 读取测试图片
#     save_path =r"C:\Users\lin\Desktop\temp\1_result.jpg"
#     img_path = r"C:\Users\lin\Desktop\141\1.jpg"   # 修改为你的图片路径
#     img = cv2.imread(img_path)
#
#     if img is None:
#         print("图像读取失败")
#         return
#     img_1 = cv2.resize(img.copy(), (2560, 1920))
#     x0, y0 = 971, 985
#     x1, y1 = 1853, 1489
#     h, w = img_1.shape[:2]
#
#     # 防止越界
#     if x1 > w or y1 > h:
#         return False, "ROI超出图像范围"
#
#     # 裁剪
#     cropped = img_1[y0:y1, x0:x1]
#
#     if cropped.size == 0:
#         return False, "裁剪失败"
#
#     # 统一尺寸（方便OCR）
#     cropped = cv2.resize(cropped, (820, 456))
#
#     # 显示提取结果
#     cv2.imshow("Screen", cropped)
#
#     # 保存结果
#     cv2.imwrite(save_path, cropped)
#
#     print("屏幕图像已保存:", save_path)
#
#     # 等待按键
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

if __name__ == "__main__":

    # ==============================
    # 1️⃣ 构造测试配置
    # ==============================
    cfg = ConfigManager(r"D:\2026\Instrument_OCR_145146\config\config.json")

     # ==============================
    # 3️⃣ 创建推理器
    # ==============================
    infer_engine = Infer(cfg)
    parser = Parser(cfg)

    # ==============================
    # 4️⃣ 测试图像路径
    # ==============================
    test_img_path = r"C:\Users\lin\Desktop\141\1.jpg"
    img = cv2.imread(test_img_path)

    # 5️⃣ 执行推理
    # ==============================
    OCR_char_results, box_img = infer_engine.infer(img)
    print(OCR_char_results)
    cv2.imshow("Screen", box_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 解析
    print(parser.parse(OCR_char_results))









