import logging
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCSUploader:
    def __init__(self, bucket_name):
        """
        Initialize the GCS uploader.

        Args:
            bucket_name (str): Name of your GCS bucket.
        """
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        logger.info(f"GCSUploader initialized for bucket: {bucket_name}")

    def upload_file(self, local_path, destination_blob):
        """
        Uploads a local file to GCS.

        Args:
            local_path (str): Path to local file to upload.
            destination_blob (str): Path in bucket (incl. folders/prefixes).
                Example: 'heatmaps/2025-07-13-heatmap.png'
        """
        blob = self.bucket.blob(destination_blob)
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {local_path} to gs://{self.bucket_name}/{destination_blob}")

    def list_files(self, prefix=None):
        """
        Lists files in the bucket (optionally filtered by prefix).

        Args:
            prefix (str): Only list blobs starting with this prefix.
        """
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
        for blob in blobs:
            print(blob.name)
