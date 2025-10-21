import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model

# ===================== KONFIGURASI DASHBOARD =====================
st.set_page_config(
    page_title="Dashboard Model - Balqis Isaura",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Dashboard Model - Balqis Isaura")
st.markdown("---")

# Sidebar untuk pilih model
model_choice = st.sidebar.radio(
    "Pilih Model:",
    ["PyTorch - YOLO", "TensorFlow - ResNet50 (Pizza / Not Pizza)"]
)

# ==================== MODEL PYTORCH YOLO ====================
if model_choice == "PyTorch - YOLO":
    st.header("🎯 Model PyTorch - YOLO")
    
    try:
        # Load YOLO model
        @st.cache_resource
        def load_yolo():
            return YOLO('model/DINI ARIFATUL NASYWA_Laporan 4.pt')
        
        with st.spinner("Loading YOLO model..."):
            model = load_yolo()
        
        st.success("✅ Model YOLO berhasil dimuat!")
        
        # Upload gambar
        st.markdown("### Upload Gambar untuk Deteksi Objek")
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=['jpg', 'jpeg', 'png'],
            key='yolo'
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)
            
            if st.button("🔍 Deteksi Objek", type="primary"):
                with st.spinner("Mendeteksi objek..."):
                    # Prediksi
                    results = model(image)
                    
                    with col2:
                        st.subheader("🎯 Hasil Deteksi")
                        result_img = results[0].plot()
                        st.image(result_img, use_column_width=True)
                    
                    # Detail deteksi
                    st.markdown("---")
                    st.subheader("📋 Detail Deteksi")
                    
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        cols = st.columns(3)
                        for i, box in enumerate(boxes):
                            col_idx = i % 3
                            with cols[col_idx]:
                                st.metric(
                                    f"Objek {i+1}",
                                    model.names[int(box.cls)],
                                    f"{box.conf[0]:.1%}"
                                )
                    else:
                        st.info("ℹ Tidak ada objek terdeteksi")
                        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Pastikan file model ada di folder 'model/'")

# ==================== MODEL TENSORFLOW ====================
elif model_choice == "TensorFlow - ResNet50 (Pizza / Not Pizza)":
    st.header("🧠 Model TensorFlow - ResNet50 (Pizza / Not Pizza)")
    
    model = None
    
    try:
        # ===================== PASTIKAN FOLDER MODEL ADA =====================
        os.makedirs("model", exist_ok=True)

        # ===================== LINK MODEL GOOGLE DRIVE =====================
        FILE_ID = "1scvZQDCuDce9rZ-WIpDW0Yl7tbYekVa7"
        MODEL_PATH = "model/model_pizza_notpizza.h5"
        DRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

        # ===================== DOWNLOAD MODEL JIKA BELUM ADA =====================
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024:
            with st.spinner("⬇ Mengunduh model dari Google Drive..."):
                gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
            st.success("✅ Model berhasil diunduh!")

        # ===================== NAMA KELAS =====================
        class_names = ["Not Pizza", "Pizza"]

        # ===================== CUSTOM LAYER (jika diperlukan) =====================
        class GetItem(Layer):
            def call(self, inputs): 
                return inputs

        # ===================== LOAD MODEL DENGAN CACHE =====================
        @st.cache_resource
        def load_tf_model():
            return load_model(MODEL_PATH, compile=False, custom_objects={'GetItem': GetItem})

        with st.spinner("🔄 Memuat model TensorFlow..."):
            model = load_tf_model()
        st.success("✅ Model TensorFlow berhasil dimuat!")

    except Exception as e:
        st.error(f"❌ Error TensorFlow: {e}")
        st.info("Pastikan file model bisa diakses dari Google Drive dan kompatibel dengan versi TensorFlow saat ini.")

    # ===================== UPLOAD & PREDIKSI GAMBAR =====================
    if model is not None:
        st.markdown("### Upload Gambar untuk Prediksi")
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=['jpg', 'jpeg', 'png'],
            key='tf'
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)
            
            if st.button("🔮 Prediksi", type="primary"):
                with st.spinner("✨ Melakukan prediksi..."):
                    # Preprocess gambar
                    img_array = np.array(image.resize((224,224))) / 255.0
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array]*3, axis=-1)
                    elif img_array.shape[-1] == 4:
                        img_array = img_array[...,:3]
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prediksi
                    predictions = model.predict(img_array, verbose=0)
                    idx = np.argmax(predictions[0])
                    predicted_class = class_names[idx]
                    confidence = predictions[0][idx]

                    with col2:
                        st.subheader("🎯 Hasil Prediksi")
                        st.metric("Kelas Prediksi", predicted_class)
                        st.metric("Confidence", f"{confidence:.2%}")

# Footer
st.markdown("---")
st.markdown("📌 Dibuat oleh Balqis Isaura | Powered by Streamlit 🚀")
