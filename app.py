import streamlit as st
import numpy as np
import cv2
from PIL import Image
import datetime
import tensorflow as tf
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Brain Tumor AI System",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection System")
st.write("Upload MRI scan for AI analysis")

# ---------------------------
# LOAD MODEL (SAFE CACHE)
# ---------------------------
@st.cache_resource
def load_model_safe():
    return tf.keras.models.load_model("tumor_model.keras", compile=False)

model = load_model_safe()

# ---------------------------
# PDF REPORT GENERATOR
# ---------------------------
def generate_pdf(result, confidence, tumor_area, tumor_pixels):
    file_name = "MRI_Report.pdf"
    c = canvas.Canvas(file_name, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(140, 750, "Brain Tumor Analysis Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 700, f"Result: {result}")
    c.drawString(50, 680, f"Confidence Score: {confidence:.4f}")
    c.drawString(50, 660, f"Tumor Area (%): {tumor_area:.2f}")
    c.drawString(50, 640, f"Tumor Pixels: {tumor_pixels}")
    c.drawString(50, 620, f"Date: {datetime.date.today()}")

    c.save()
    return file_name

# ---------------------------
# UPLOAD IMAGE
# ---------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("L")
    image = np.array(image)

    st.subheader("Original MRI")
    st.image(image, width=300)

    # ---------------------------
    # PREPROCESS
    # ---------------------------
    img = cv2.resize(image, (128, 128)) / 255.0
    img_input = np.expand_dims(img, axis=(0, -1))

    # ---------------------------
    # PREDICTION
    # ---------------------------
    pred = model.predict(img_input)[0]

    # SAFE FIX (IMPORTANT)
    if len(pred.shape) == 3:
        pred = np.squeeze(pred)

    pred = cv2.GaussianBlur(pred, (3, 3), 0)

    # ---------------------------
    # THRESHOLD MASK
    # ---------------------------
    threshold = 0.3
    mask = (pred > threshold).astype(np.uint8)

    # resize back
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # ---------------------------
    # METRICS (FIXED LOGIC)
    # ---------------------------
    tumor_pixels = np.sum(mask)
    total_pixels = mask.size

    tumor_area = (tumor_pixels / total_pixels) * 100

    max_prob = np.max(pred)
    confidence = float((tumor_area / 100) * 0.7 + max_prob * 0.3)

    # ---------------------------
    # FINAL DECISION (IMPORTANT FIX)
    # ---------------------------
    if tumor_area > 1.0 or confidence > 0.25:
        result = "Tumor Detected"
        st.error("🧠 Tumor Detected")
    else:
        result = "No Tumor Detected"
        st.success("✅ No Tumor Detected")

    st.write(f"Confidence Score: {confidence:.4f}")

    # ---------------------------
    # MASK DISPLAY
    # ---------------------------
    st.subheader("Tumor Mask")
    st.image(mask * 255, width=300)

    # ---------------------------
    # OVERLAY
    # ---------------------------
    overlay = cv2.addWeighted(image, 0.7, mask * 255, 0.3, 0)

    st.subheader("Overlay View")
    st.image(overlay, width=300)

    # ---------------------------
    # STATS
    # ---------------------------
    st.subheader("Tumor Statistics")
    st.write(f"Tumor Pixels: {tumor_pixels}")
    st.write(f"Tumor Area: {tumor_area:.2f}%")

    # ---------------------------
    # GRAPH
    # ---------------------------
    st.subheader("Confidence Graph")
    fig, ax = plt.subplots()
    ax.bar(["Confidence", "Threshold"], [confidence, 0.25])
    st.pyplot(fig)

    # ---------------------------
    # PDF DOWNLOAD
    # ---------------------------
    if st.button("📄 Generate Medical Report"):
        file = generate_pdf(result, confidence, tumor_area, tumor_pixels)
        st.success("Report Generated!")

        with open(file, "rb") as f:
            st.download_button(
                "Download PDF Report",
                f,
                file_name="MRI_Report.pdf"
            )
