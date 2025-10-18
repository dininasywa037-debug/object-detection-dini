import os
import streamlit as st
from PIL import Image
import numpy as np

# ===========================
# Konfigurasi YOLO
# ===========================
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.makedirs("/tmp/Ultralytics", exist_ok=True)

# ===========================
# Sidebar
# ===========================
st.title("🧠 YOLO & TensorFlow Image App")
menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# ===========================
# Load Model YOLO
# ===========================
@st.cache_resource
def load_yolo_model():
    from ultralytics import YOLO
    model = YOLO("model/DINI ARIFATUL_NASYWA_Laporan 4.pt")
    return model

# ===========================
# Load Model TensorFlow
# ===========================
@st.cache_resource
def load_classifier_model():
    import tensorflow as tf
    model = tf.keras.models.load_model("Dini_Arifatul_Nasywa_laporan2.h5")
    return model

# ===========================
# Proses Upload
# ===========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        st.write("🔍 Mendeteksi objek...")
        yolo_model = load_yolo_model()
        img_array = np.array(img)
        results = yolo_model(img_array)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
        st.success("✅ Deteksi selesai!")

    elif menu == "Klasifikasi Gambar":
        st.write("🔍 Memprediksi kelas gambar...")
        classifier = load_classifier_model()
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        class_names = ["Kelas1", "Kelas2", "Kelas3"]  # Ganti sesuai model
        st.write("### Hasil Prediksi:", class_names[class_index])
        st.write("Probabilitas:", np.max(prediction))

else:
    st.info("Silakan unggah gambar terlebih dahulu.")
