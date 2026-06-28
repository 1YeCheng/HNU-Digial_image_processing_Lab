import cv2

class MP4Camera:
    def __init__(self, mp4_path, loop=True):
        self.mp4_path = mp4_path
        self.loop = loop
        self.cap = None

    def connect(self):
        self.cap = cv2.VideoCapture(self.mp4_path)
        return self.cap.isOpened()

    def reconnect(self):
        self.close()
        return self.connect()

    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        if self.cap:
            self.cap.release()