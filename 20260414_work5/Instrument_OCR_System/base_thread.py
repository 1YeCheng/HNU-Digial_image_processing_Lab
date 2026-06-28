import threading
import logging

class BaseThread(threading.Thread):
    def __init__(self, name="BaseThread"):
        super().__init__(name=name, daemon=True)
        self._running = True
        self._healthy = True

    def run(self):
        try:
            self._on_start()
            self._run()
        except Exception as e:
            logging.exception(f"线程[{self.name}]异常: {e}")
            self._healthy = False
        finally:
            self._on_stop()

    def _run(self):
        pass

    def _on_start(self):
        logging.info(f"线程[{self.name}]已启动")

    def _on_stop(self):
        logging.info(f"线程[{self.name}]已停止")

    def stop(self):
        self._running = False

    def is_running(self):
        return self._running

    def is_healthy(self):
        return self._healthy