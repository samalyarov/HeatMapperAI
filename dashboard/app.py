import streamlit as st
from google.cloud import storage
from dotenv import load_dotenv
import os
from io import BytesIO
from PIL import Image

# ✅ Load .env
load_dotenv()
BUCKET_NAME = os.getenv('BUCKET_NAME')

# ✅ Initialize GCS client
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

def list_heatmaps():
    """
    List all heatmap images in GCS bucket under heatmaps/ prefix.
    """
    blobs = bucket.list_blobs(prefix='heatmaps/')
    files = [blob.name for blob in blobs if blob.name.endswith('.png')]
    files.sort(reverse=True)  # Latest first
    return files

def download_image(blob_name):
    """
    Download image blob as PIL.Image.
    """
    blob = bucket.blob(blob_name)
    data = blob.download_as_bytes()
    image = Image.open(BytesIO(data))
    return image

# ✅ Streamlit App
st.set_page_config(page_title="HeatMapperAI Dashboard", layout="centered")
st.title("🚗 HeatMapperAI Dashboard")
st.markdown("View generated heatmaps stored in Google Cloud Storage.")

# ✅ Sidebar: list heatmaps
st.sidebar.header("Available Heatmaps")

heatmap_files = list_heatmaps()

if not heatmap_files:
    st.warning("No heatmaps found in bucket!")
else:
    selected_file = st.sidebar.selectbox("Choose Heatmap", heatmap_files)
    
    # ✅ Download & show
    with st.spinner("Loading heatmap..."):
        image = download_image(selected_file)
        st.image(image, caption=selected_file, use_column_width=True)
        st.success(f"Displaying: {selected_file}")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by HeatMapperAI")
