import cv2
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraStream:
    def __init__(self, url):
        """
        Initialize the camera stream.

        Args:
            url (str): MJPEG stream URL or local video path.
        """
        self.url = url
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video stream at {url}")
        logger.info(f"Opened video stream: {url}")

    def get_frame(self):
        """
        Grab a single frame from the stream.

        Returns:
            frame (numpy.ndarray): BGR image.
        """
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to grab frame.")
            return None
        return frame

    def frame_generator(self, interval=5):
        """
        Generator that yields frames every 'interval' seconds.

        Args:
            interval (int): Number of seconds to wait between frames.
        """
        try:
            while True:
                frame = self.get_frame()
                if frame is not None:
                    yield frame
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Stream interrupted by user.")
        finally:
            self.release()

    def release(self):
        """
        Release the video capture resource.
        """
        self.cap.release()
        logger.info("Video stream released.")
