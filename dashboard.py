import os
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# ==========================================
# Konfigurasi YOLO agar tidak error
# ==========================================
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# ==========================================
# Sidebar & Upload
# ==========================================
st.title("🧠 Image Detection & Classification App")
menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    # ==============================
    # Mode Deteksi YOLO
    # ==============================
    if menu == "Deteksi Objek (YOLO)":
        st.write("🔍 Mendeteksi objek...")

        from ultralytics import YOLO

        @st.cache_resource
        def load_yolo():
            model = YOLO("model/DINI ARIFATUL NASYWA_Laporan 4.pt")
            return model

        yolo_model = load_yolo()
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
        st.success("✅ Deteksi selesai!")

    # ==============================
    # Mode Klasifikasi TensorFlow
    # ==============================
    elif menu == "Klasifikasi Gambar":
        st.write("🔍 Memprediksi kelas gambar...")

        import tensorflow as tf
        from tensorflow.keras.preprocessing import image

        @st.cache_resource
        def load_classifier():
            model = tf.keras.models.load_model("Dini_Arifatul_Nasywa_laporan2.h5")
            return model

        classifier = load_classifier()

        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        st.write("### Hasil Prediksi:", class_index)
        st.write("Probabilitas:", np.max(prediction))

else:
    st.info("Silakan unggah gambar terlebih dahulu.")
