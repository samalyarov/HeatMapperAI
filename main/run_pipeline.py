import cv2
import time
import datetime
import csv
import os
import logging
from dotenv import load_dotenv

from camera.stream_reader import CameraStream
from detection.yolo_infer import YOLODetector
from heatmap.heatmap_generator import HeatmapGenerator
from storage.gcs_uploader import GCSUploader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ CONFIG
load_dotenv()
STREAM_URL = os.getenv('STREAM_URL')
BACKGROUND_IMAGE_PATH = os.getenv('BACKGROUND_IMAGE_PATH')
BUCKET_NAME = os.getenv('BUCKET_NAME')
LOCAL_OUTPUT_DIR = os.getenv('LOCAL_OUTPUT_DIR')
FRAME_INTERVAL = int(os.getenv('FRAME_INTERVAL', 5))
HEATMAP_INTERVAL = int(os.getenv('HEATMAP_INTERVAL', 30))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.3))
MODEL_PATH = os.getenv('MODEL_PATH')


def save_detections_csv(points, output_csv):
    """
    Saves detection centers as CSV file.
    """
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])
        writer.writerows(points)
    logger.info(f"Saved detections CSV to {output_csv}")

def main():
    # ✅ Make local output folder
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    # ✅ Load background image for heatmaps
    background_bgr = cv2.imread(BACKGROUND_IMAGE_PATH)
    if background_bgr is None:
        logger.error(f"Background image not found at {BACKGROUND_IMAGE_PATH}")
        return
    background_rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)

    # ✅ Initialize components
    camera = CameraStream(STREAM_URL)
    detector = YOLODetector(model_path=MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD)
    heatmap_gen = HeatmapGenerator(background_rgb, alpha=0.4)
    uploader = GCSUploader(BUCKET_NAME)

    # ✅ In-memory list of detection points
    all_detections = []

    try:
        logger.info("Starting pipeline loop...")
        for frame in camera.frame_generator(interval=FRAME_INTERVAL):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # ✅ Run detection
            detections = detector.detect(frame)
            logger.info(f"Frame {timestamp}: {len(detections)} detections.")

            # ✅ Add center points
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                all_detections.append((x_center, y_center))

            # ✅ Every N detections, generate heatmap
            if len(all_detections) >= HEATMAP_INTERVAL:
                heatmap_filename = f"heatmap_{timestamp}.png"
                heatmap_path = os.path.join(LOCAL_OUTPUT_DIR, heatmap_filename)

                # Generate and save
                heatmap_gen.generate_heatmap(all_detections, output_path=heatmap_path)
                logger.info(f"Generated heatmap: {heatmap_path}")

                # Upload heatmap to GCS
                uploader.upload_file(
                    local_path=heatmap_path,
                    destination_blob=f"heatmaps/{heatmap_filename}"
                )

                # Also save and upload CSV of detections
                csv_filename = f"detections_{timestamp}.csv"
                csv_path = os.path.join(LOCAL_OUTPUT_DIR, csv_filename)
                save_detections_csv(all_detections, csv_path)

                uploader.upload_file(
                    local_path=csv_path,
                    destination_blob=f"detections/{csv_filename}"
                )

                # Reset buffer
                all_detections.clear()

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user. Exiting gracefully...")
    finally:
        camera.release()

if __name__ == "__main__":
    main()
