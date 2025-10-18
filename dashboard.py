import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ==========================================
# FIX: Konfigurasi YOLO agar tidak error di Streamlit Cloud
# ==========================================
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# ==========================================
# Load Model YOLO
# ==========================================
@st.cache_resource
def load_yolo():
    model = YOLO("model/DINI ARIFATUL NASYWA_Laporan 4.pt")
    return model

yolo_model = load_yolo()

# ==========================================
# UI
# ==========================================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        st.write("🔍 Mendeteksi objek...")

        # Jalankan deteksi
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi dengan bounding box

        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
        st.success("✅ Deteksi selesai!")

else:
    st.info("Silakan unggah gambar terlebih dahulu.")
