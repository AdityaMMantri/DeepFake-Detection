import streamlit as st
from PIL import Image
import numpy as np
import os

from load_model import load_model
from predict import predict


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="DeepFake Detector",
    layout="centered"
)

st.title("DeepFake Detection System")


# =========================================================
# MODEL SELECTION
# =========================================================
model_type = st.selectbox(
    "Select Model",
    ["cnn", "vit_single", "vit_9ch"]
)


# =========================================================
# FORCE RELOAD BUTTON (IMPORTANT)
# =========================================================
if st.button("🔄 Reload Model"):
    st.cache_resource.clear()
    st.success("Cache cleared. Model will reload.")


# =========================================================
# LOAD MODEL (cached)
# =========================================================
@st.cache_resource
def get_model(model_type):
    st.write(f"[LOADING MODEL]: {model_type}")
    return load_model(model_type)


model = get_model(model_type)


# =========================================================
# SHOW MODEL INFO (UI DEBUG)
# =========================================================
st.markdown("### 🔍 Model Debug Info")

MODEL_PATHS = {
    "cnn": r"F:\SEM-6\DL\DEEP-FAKE\DL-Project\checkpoints\cnn\best_model.pth",
    "vit_9ch": r"F:\SEM-6\DL\DEEP-FAKE\DL-Project\checkpoints\vit\best_model.pth",
    "vit_single": r"F:\SEM-6\DL\DEEP-FAKE\DL-Project\checkpoints\vit_single\best_auc.pth"
}

full_path = os.path.abspath(MODEL_PATHS[model_type])

st.write(f"**Selected Model:** `{model_type}`")
st.write(f"**Checkpoint Path:** `{full_path}`")
st.write(f"**Exists:** `{os.path.exists(full_path)}`")
st.write(f"**Active Class:** `{type(model).__name__}`")


# =========================================================
# INPUT METHOD
# =========================================================
input_type = st.radio(
    "Choose Input Method",
    ["Upload Image", "Use Camera"]
)

image = None

if input_type == "Upload Image":
    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

elif input_type == "Use Camera":
    captured = st.camera_input("Capture Image")

    if captured:
        image = Image.open(captured).convert("RGB")


# =========================================================
# DISPLAY IMAGE
# =========================================================
if image:
    st.image(image, caption="Input Image", use_column_width=True)


# =========================================================
# RUN INFERENCE
# =========================================================
if image and st.button("🚀 Run Detection"):

    result = predict(model, image, model_type)

    st.markdown("---")
    st.subheader("📊 Result")

    st.write(f"**Prediction:** {result['prediction'].upper()}")
    st.write(f"**Confidence:** {result['confidence']:.4f}")
    st.write(f"**Fake Probability:** {result['p_fake']:.4f}")

    # =====================================================
    # EXTRA (for VIT 9ch)
    # =====================================================
    if result.get("extra"):
        st.markdown("### 🔬 Advanced Analysis")

        extra = result["extra"]

        if extra.get("p_vit") is not None:
            st.write(f"ViT Raw Probability: {extra['p_vit']:.4f}")

        if extra.get("trust_score") is not None:
            st.write(f"Trust Score: {extra['trust_score']:.4f}")

        if extra.get("explanation"):
            st.markdown("**Explanation:**")
            # Convert bullet prefix characters so Streamlit renders them as markdown lists
            explanation_md = extra["explanation"].replace("• ", "- ").replace("→ ", "> ")
            st.markdown(explanation_md)