"""
ClearScan AI — interactive demo.

Upload a degraded semiconductor inspection image and view the restored output
side by side, along with PSNR/SSIM if a ground-truth clean image is also provided.

Run:
    streamlit run demo/app.py
"""
import io
import os
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import UNetRestorer  # noqa: E402

st.set_page_config(page_title="ClearScan AI", layout="wide")
st.title("🔬 ClearScan AI")
st.caption("AI-Based Restoration of Degraded Images for Semiconductor Inspection")

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "clearscan.pt")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetRestorer().to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        st.sidebar.success("Loaded trained checkpoint")
    else:
        st.sidebar.warning("No checkpoint found — using untrained weights (demo only). Train a model first for real results.")
    model.eval()
    return model, device


def preprocess(img_pil, size=256):
    img = img_pil.convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def postprocess(tensor, size):
    arr = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).resize(size)


model, device = load_model()

col1, col2 = st.columns(2)
with col1:
    degraded_file = st.file_uploader("Upload degraded image", type=["png", "jpg", "jpeg"])
with col2:
    clean_file = st.file_uploader("Optional: ground-truth clean image (for PSNR/SSIM)", type=["png", "jpg", "jpeg"])

if degraded_file:
    degraded_img = Image.open(degraded_file)
    input_tensor = preprocess(degraded_img).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    restored_img = postprocess(output_tensor, degraded_img.size)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Degraded Input")
        st.image(degraded_img, use_column_width=True)
    with c2:
        st.subheader("Restored Output")
        st.image(restored_img, use_column_width=True)

    if clean_file:
        clean_img = Image.open(clean_file).convert("RGB").resize(degraded_img.size)
        clean_arr = np.array(clean_img).astype(np.float32) / 255.0
        restored_arr = np.array(restored_img).astype(np.float32) / 255.0

        p = psnr(clean_arr, restored_arr, data_range=1.0)
        s = ssim(clean_arr, restored_arr, data_range=1.0, channel_axis=2)

        m1, m2 = st.columns(2)
        m1.metric("PSNR", f"{p:.2f} dB")
        m2.metric("SSIM", f"{s:.4f}")

    buf = io.BytesIO()
    restored_img.save(buf, format="PNG")
    st.download_button("Download restored image", buf.getvalue(), file_name="restored.png", mime="image/png")
else:
    st.info("Upload a degraded semiconductor inspection image to see the restoration in action.")
