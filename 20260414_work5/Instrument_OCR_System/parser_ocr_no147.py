class Parser:
    def __init__(self, config):
        self.config = config

    def parse(self, ocr_results):
        temperature = None
        pressure = None
        flow = None
        total = None

        # 从识别结果里挨个找
        for res in ocr_results:
            text = str(res.content).lower()

            if "temp" in text or "温度" in text:
                temperature = self._get_number(text)
            elif "press" in text or "压力" in text:
                pressure = self._get_number(text)
            elif "flow" in text or "流量" in text:
                flow = self._get_number(text)
            elif "total" in text or "累计" in text:
                total = self._get_number(text)

        return [{
            "temperature": temperature,
            "pressure": pressure,
            "flow": flow,
            "total": total
        }]

    # 从文字里抽取数字
    def _get_number(self, text):
        for word in text.split():
            try:
                return float(word)
            except:
                continue
        return None