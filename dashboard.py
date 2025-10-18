import os
import streamlit as st
from PIL import Image
import numpy as np

# ==========================================
# Konfigurasi YOLO
# ==========================================
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.makedirs("/tmp/Ultralytics", exist_ok=True)

st.title("🧠 YOLO Object Detection App")

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    st.write("🔍 Mendeteksi objek...")

    from ultralytics import YOLO

    # ===============================
    # Load model (cached supaya cepat)
    # ===============================
    @st.cache_resource
    def load_yolo_model():
        model = YOLO("model/DINI ARIFATUL_NASYWA_Laporan 4.pt")
        return model

    yolo_model = load_yolo_model()

    # Convert PIL ke ndarray
    img_array = np.array(img)
    results = yolo_model(img_array)

    # Tampilkan hasil
    result_img = results[0].plot()
    st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
    st.success("✅ Deteksi selesai!")

else:
    st.info("Silakan unggah gambar terlebih dahulu.")
