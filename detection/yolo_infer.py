from ultralytics import YOLO
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COCO class IDs for vehicle types
DEFAULT_VEHICLE_CLASSES = {
    2: 'Car',
    3: 'Motorcycle',
    5: 'Bus',
    7: 'Truck'
}

class YOLODetector:
    def __init__(self, model_path='yolov8l.pt', target_classes=None, conf_threshold=0.3):
        """
        Initialize YOLO detector.

        Args:
            model_path (str): Path to YOLO model weights.
            target_classes (dict): Class ID to label mapping.
            conf_threshold (float): Minimum confidence threshold.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes if target_classes else DEFAULT_VEHICLE_CLASSES
        logger.info(f"YOLO model loaded from {model_path}")
        logger.info(f"Target classes: {self.target_classes}")

    def detect(self, frame):
        """
        Run YOLO detection on an input frame.

        Args:
            frame (numpy.ndarray): BGR image.

        Returns:
            detections (list of dict): Each dict has class_id, label, conf, bbox.
        """
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id in self.target_classes and conf >= self.conf_threshold:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    detection = {
                        'class_id': cls_id,
                        'label': self.target_classes[cls_id],
                        'conf': conf,
                        'bbox': xyxy
                    }
                    detections.append(detection)

        logger.info(f"Detected {len(detections)} objects")
        return detections
