"""DeepVision proof-of-concept UI: requires a compatible trained checkpoint."""
import tempfile
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from deepvision.inference import load_model, normalized_heatmap, predict

st.set_page_config(page_title="DeepVision Crowd Monitor", page_icon="📹", layout="wide")
st.title("DeepVision Crowd Monitor")
st.caption("Research prototype — crowd density estimation and threshold-based alerts")
st.warning("No pretrained model is included. Do not interpret outputs as reliable crowd counts without training and validation.")
checkpoint = st.file_uploader("Upload a compatible, trusted CSRNet .pth/.pt checkpoint", type=["pth", "pt"])
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
threshold = st.number_input("Overcrowding count threshold (demo only)", min_value=1, value=100)

if st.button("Estimate crowd"):
    if checkpoint is None or image_file is None:
        st.error("Both a trained checkpoint and an image are required. Random weights are not used.")
    else:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
                tmp.write(checkpoint.getvalue())
                temp_path = tmp.name
            model = load_model(temp_path)
            image = Image.open(image_file).convert("RGB")
            density, count = predict(model, image)
            col1, col2 = st.columns(2)
            col1.image(image, caption="Input image", use_container_width=True)
            heat = cv2.applyColorMap(normalized_heatmap(density), cv2.COLORMAP_JET)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            col2.image(heat, caption="Estimated density heatmap", use_container_width=True)
            st.metric("Estimated crowd count", f"{count:.1f}")
            if count >= threshold:
                st.error("Threshold exceeded — demo alert")
            else:
                st.success("Below configured threshold — demo status")
        except Exception as exc:
            st.error(f"Inference failed: {exc}. Confirm checkpoint architecture and preprocessing.")
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)
