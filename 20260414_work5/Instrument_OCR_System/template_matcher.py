import os
import cv2
import numpy as np
import logging

class TemplateMatcher:
    def __init__(self, template_dir, img_size=(40, 60), method="template", threshold=0.5):
        self.img_size = img_size
        self.threshold = threshold
        self.templates = {}
        self._load_dummy_templates()

    def _load_dummy_templates(self):
        self.templates = {
            "temp": [np.zeros((60, 40), np.uint8)],
            "press": [np.zeros((60, 40), np.uint8)],
            "flow": [np.zeros((60, 40), np.uint8)],
            "total": [np.zeros((60, 40), np.uint8)],
            "0": [np.zeros((60, 40), np.uint8)],
            "1": [np.zeros((60, 40), np.uint8)],
            "2": [np.zeros((60, 40), np.uint8)],
            "3": [np.zeros((60, 40), np.uint8)],
            "4": [np.zeros((60, 40), np.uint8)],
            "5": [np.zeros((60, 40), np.uint8)],
        }

    def _standardize(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def predict(self, img, return_topk=1):
        img = self._standardize(img)
        try:
            img = cv2.resize(img, (40, 60))
        except:
            return "unknown", 0

        best_label = "unknown"
        best_score = 0

        for label, templ_list in self.templates.items():
            for t in templ_list:
                try:
                    res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
                    score = float(res.max())
                    if score > best_score:
                        best_score = score
                        best_label = label
                except:
                    continue

        if best_score < 0.3:
            return "unknown", best_score
        return best_label, best_score