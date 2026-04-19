import streamlit as st
import numpy as np
import cv2
from PIL import Image
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import tensorflow as tf

# ---------------------------
# LOAD MODEL (SAFE)
# ---------------------------
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("tumor_model.keras", compile=False)

model = load_my_model()

# ---------------------------
# UI SETUP
# ---------------------------
st.set_page_config(page_title="MRI Tumor AI System", layout="centered")

st.title("🧠 MRI Brain Tumor Detection System")
st.write("Upload MRI scan for AI analysis")

# ---------------------------
# PDF REPORT
# ---------------------------
def generate_pdf(result, confidence, tumor_area, tumor_pixels):
    file_name = "MRI_Report.pdf"
    c = canvas.Canvas(file_name, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(150, 750, "MRI Brain Tumor Analysis Report")

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

    # Read image
    image = Image.open(uploaded_file).convert("L")
    image = np.array(image)

    st.subheader("Original MRI")
    st.image(image, width=300)

    # ---------------------------
    # PREPROCESS
    # ---------------------------
    img = cv2.resize(image, (128, 128))
    img = img / 255.0
    img_input = np.expand_dims(img, axis=(0, -1))

    # ---------------------------
    # PREDICTION
    # ---------------------------
    pred = model.predict(img_input)[0]
    pred = cv2.GaussianBlur(pred, (3, 3), 0)

    # ---------------------------
    # METRICS
    # ---------------------------
    tumor_pixels = np.sum(pred > 0.3)
    total_pixels = pred.size
    tumor_ratio = tumor_pixels / total_pixels

    max_prob = np.max(pred)
    confidence = (tumor_ratio * 0.7) + (max_prob * 0.3)

    # ---------------------------
    # RESULT
    # ---------------------------
    st.subheader("Diagnosis Result")

    if confidence > 0.02:
        result = "Tumor Detected"
        st.error("🧠 Tumor Detected")
    else:
        result = "No Tumor Detected"
        st.success("✅ No Tumor Detected")

    st.write(f"Confidence Score: {confidence:.4f}")

    # ---------------------------
    # MASK
    # ---------------------------
    mask = (pred > 0.3).astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

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
    st.write(f"Tumor Area: {tumor_ratio * 100:.2f}%")

    # ---------------------------
    # GRAPH
    # ---------------------------
    st.subheader("Confidence Graph")

    fig, ax = plt.subplots()
    ax.bar(["Confidence", "Threshold"], [confidence, 0.02])
    st.pyplot(fig)

    # ---------------------------
    # PDF DOWNLOAD
    # ---------------------------
    if st.button("📄 Generate Medical Report"):
        file = generate_pdf(result, confidence, tumor_ratio * 100, tumor_pixels)
        st.success("Report Generated!")

        with open(file, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="MRI_Report.pdf")
