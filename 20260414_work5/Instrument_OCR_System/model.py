class OCRCharResult:
    def __init__(self, x, y, w, h, content, confidence):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.content = content
        self.confidence = confidence