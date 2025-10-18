import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# ==========================
# Load Models
# ==========================
@st.cache_resource(show_spinner=True, allow_output_mutation=True)
def load_models():
    # Gunakan Path agar path file tetap valid di berbagai sistem
    yolo_path = Path("model/DINI ARIFATUL NASYWA_Laporan 4.pt")
    classifier_path = Path("Dini_Arifatul_Nasywa_laporan2.h5")
    
    # Load kedua model
    yolo_model = YOLO(str(yolo_path))
    classifier = tf.keras.models.load_model(str(classifier_path))
    return yolo_model, classifier

# Panggil fungsi load model
yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_column_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (NumPy array BGR)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.image(result_img, caption="Hasil Deteksi", use_column_width=True)

    elif menu == "Klasifikasi Gambar":
        # Preprocessing
        img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        # Opsional: tambahkan nama label kalau ada
        class_labels = ["Kelas 1", "Kelas 2", "Kelas 3"]  # ubah sesuai dataset kamu
        
        st.write("### Hasil Prediksi:")
        st.success(f"{class_labels[class_index] if class_index < len(class_labels) else class_index}")
        st.write(f"**Probabilitas:** {confidence:.2f}")

