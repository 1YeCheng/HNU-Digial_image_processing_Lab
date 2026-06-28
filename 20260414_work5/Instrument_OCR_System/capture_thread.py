import time
import logging
from queue import Queue
from base_thread import BaseThread
from ConfigManager import ConfigManager
import cv2

class CaptureThread(BaseThread):
    """
    工业级视频采集线程（继承 BaseThread）

    - 每 interval_sec 秒采集 frames_per_batch 帧
    - 每帧生成 frame_id
    - 放入 image_queue
    - 异常自动交给 BaseThread 处理
    """

    def __init__(
        self,
        image_queue: Queue,
        config,
            cam,
        name="CaptureThread"
    ):
        super().__init__(name=name)

        self.image_queue = image_queue
        self.config = config
        self.cam = cam
        self.frames_per_batch = config.get_config().get("camera").get("frames_per_batch")
        self.interval_sec = config.get_config().get("camera").get("sample_interval")
        self.frame_counter = 0

    # ==============================
    # 真正执行逻辑
    # ==============================
    def _run(self):

        logging.info("[CaptureThread] 启动")

        if not self.cam.connect():
            raise RuntimeError("摄像头连接失败")

        try:
            while self.is_running():

                start_time = time.perf_counter()

                for _ in range(self.frames_per_batch):

                    if not self.is_running():
                        break

                    frame = self.cam.read_frame()

                    if frame is None:
                        logging.warning("[CaptureThread] 读取失败，尝试重连")
                        self.cam.reconnect()
                        continue

                    timestamp = int(time.time() * 1000)
                    frame_id = f"{timestamp}_{self.frame_counter}"
                    self.frame_counter += 1

                    frame_item = {
                        "frame_id": frame_id,
                        "image": frame
                    }

                    try:
                        self.image_queue.put_nowait(frame_item)
                    except:
                        logging.warning("[CaptureThread] 队列已满，丢弃帧")

                # 精确间隔控制
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, self.interval_sec - elapsed)

                while self.is_running() and sleep_time > 0:
                    step = min(0.2, sleep_time)
                    time.sleep(step)
                    sleep_time -= step

        finally:
            self.cam.close()
            logging.info("[CaptureThread] 停止")

    # ==============================
    # 可选：重写健康判断（可扩展）
    # ==============================
    def is_healthy(self):
        return super().is_healthy()

#
# def main():
#     frame_queue = deque(maxlen=100)
#     queue_lock = threading.Lock()
#
#     # 创建 MP4 视频模拟摄像头
#     cam = MP4Camera(mp4_path="D:/2026/instrument_reading/mp4/1.mp4", name="TestMP4", loop=True)
#
#     # 视频线程：每隔 60 秒读取 3 帧
#     video_thread = CaptureThread(cam=cam,
#                                       frame_queue=frame_queue,
#                                       queue_lock=queue_lock,
#                                       frames_per_batch=3,
#                                       interval_sec=60)
#     video_thread.start()
#
#     try:
#         while True:
#             with queue_lock:
#                 while frame_queue:
#                     frame = frame_queue.popleft()
#                     # 显示或处理帧
#                     cv2.imshow("Video Batch Frame", frame.get("image"))
#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         raise KeyboardInterrupt
#             time.sleep(0.1)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         video_thread.stop()
#         cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()
