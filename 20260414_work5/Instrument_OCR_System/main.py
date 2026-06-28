import logging
import time
from queue import Queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def main():
    print("=" * 60)
    print("      基于机器视觉的工业仪表屏幕自动识别系统")
    print("=" * 60)

    from ConfigManager import ConfigManager
    from mp4_cam import MP4Camera
    from capture_thread import CaptureThread
    from data_process_thread import DataProcessThread
    from infer import Infer
    from parser_ocr_no147 import Parser

    config = ConfigManager("config.json")

    image_queue = Queue(maxsize=30)
    file_storage_queue = Queue(maxsize=30)
    modbus_queue = Queue(maxsize=10)

    cam = MP4Camera("test.mp4", loop=True)
    infer = Infer(config)
    parser = Parser(config)

    capture_thread = CaptureThread(image_queue, config, cam)
    process_thread = DataProcessThread(
        image_queue=image_queue,
        file_storage_queue=file_storage_queue,
        infer=infer,
        parser=parser,
        modbus_queue=modbus_queue,
        config=config
    )

    capture_thread.start()
    process_thread.start()
    logging.info("✅ 系统启动成功，正在识别...")

    try:
        while True:
            if not modbus_queue.empty():
                result = modbus_queue.get()
                print("\n📊 识别结果：")
                print(f"  温度: {result.get('temperature')}")
                print(f"  压力: {result.get('pressure')}")
                print(f"  流量: {result.get('flow')}")
                print(f"  累计: {result.get('total')}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n🛑 正在停止系统...")
    finally:
        capture_thread.stop()
        process_thread.stop()
        time.sleep(1)
        print("✅ 系统已安全停止")

if __name__ == "__main__":
    main()