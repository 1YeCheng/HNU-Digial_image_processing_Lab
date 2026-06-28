from queue import Empty
import time
from base_thread import BaseThread
import logging

class DataProcessThread(BaseThread):
    """
    数据处理线程（工业级标准版）

    功能：
    - 从 image_queue 获取 batch 帧
    - 调用 Infer + Parser
    - 多帧融合
    - 推送最大值到 modbus_queue
    """

    def __init__(
        self,
        image_queue,
        file_storage_queue,
        infer,
        parser,
        modbus_queue,
        config,
        name="DataProcessThread"
    ):
        super().__init__(name=name)

        self.image_queue = image_queue
        self.file_storage_queue = file_storage_queue
        self.infer = infer
        self.parser = parser
        self.modbus_queue = modbus_queue
        self.config = config

        # 读取一次配置
        cfg = self.config.get_config()
        self.batch_size = cfg.get("camera", {}).get("sample_count", 1)

        self.last_values = {
            "temperature": None,
            "pressure": None,
            "flow": None,
            "total": None
        }

    # ==============================
    # 生命周期管理
    # ==============================

    def _on_start(self):
        logging.info("DataProcessThread started")

    def _on_stop(self):
        logging.info("DataProcessThread stopped")

    # ==============================
    # 主循环
    # ==============================
    def _run(self):
        """
        重写 BaseThread 的抽象方法
        """
        while self._running:  # 假设 BaseThread 有 _running 标志
            self._run_loop()

    def _run_loop(self):

        frames = []

        # 1️⃣ 采集 batch
        try:
            for _ in range(self.batch_size):
                frame = self.image_queue.get(timeout=2)
                frames.append(frame)
        except Empty:
            return  # 没取够直接下一轮

        batch_results = []

        # 2️⃣ 推理 + 解析
        for frame in frames:

            frame_id = frame.get("frame_id")
            image = frame.get("image")

            if image is None:
                continue

            try:
                ocr_results, draw_img = self.infer.infer(image)
                parsed_results = self.parser.parse(ocr_results)

            except Exception as e:
                logging.exception(f"infer/parse error: {e}")
                continue

            frame_item = {
                "frame_id": frame_id,
                "image": draw_img,
                "result": parsed_results
            }

            try:
                self.file_storage_queue.put_nowait(frame_item)
            except:
                pass

            batch_results.append(parsed_results)

        # 3️⃣ 多帧融合
        fused_result = self._fuse(batch_results)
        if fused_result:
            self.last_values.update(fused_result)

            try:
                self.modbus_queue.put_nowait(fused_result)
            except:
                pass

        time.sleep(0.01)

    # ==============================
    # 多帧融合逻辑
    # ==============================

    def _fuse(self, batch_results):

        valid_values = {
            "temperature": [],
            "pressure": [],
            "flow": [],
            "total": []
        }

        for frame_result in batch_results:

            if not frame_result:
                continue

            for item in frame_result:
                for k, v in item.items():
                    if not k or v is None:
                        continue
                    valid_values[k].append(v)

        fused = {}

        for key in valid_values:
            if valid_values[key]:
                fused[key] = max(valid_values[key])
            else:
                fused[key] = self.last_values.get(key)

        if all(v is None for v in fused.values()):
            return None

        return fused
