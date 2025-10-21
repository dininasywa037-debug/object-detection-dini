import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from PIL import Image
import numpy as np
from ultralytics import YOLO
import gdown
import os
import h5py

# ===================== KONFIGURASI DASHBOARD =====================
st.set_page_config(
    page_title="Dashboard Model - Balqis Isaura",
    page_icon="✨",
    layout="wide"
)

# ===================== STYLE PINK THEME =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
[data-testid="stAppViewContainer"] { background-color: #fff5f8 !important; font-family: 'Poppins', sans-serif; }
[data-testid="stHeader"] { background-color: transparent !important; }
/* ...sisa CSS seperti sebelumnya tetap sama... */
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown("# ✨ DASHBOARD MODEL ✨")
st.markdown('<p class="subtitle">GROUP PROJECT PRESENTATION - BALQIS ISAURA</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

# ===================== PROCESS 1 - YOLO =====================
with col1:
    st.markdown("""<div class="process-card"><h2>📸 PROCESS 1</h2><h3>YOLO Object Detection</h3><div class="process-card-content">""", unsafe_allow_html=True)
    try:
        @st.cache_resource
        def load_yolo_model():
            return YOLO("model/DINI ARIFATUL NASYWA_Laporan 4.pt")
        with st.spinner("🔄 Memuat model YOLO..."):
            yolo_model = load_yolo_model()
        st.success("✅ Model YOLO berhasil dimuat!")

        uploaded_file_yolo = st.file_uploader("Upload gambar untuk deteksi objek:", type=["jpg","jpeg","png"], key="yolo")
        if uploaded_file_yolo:
            img = Image.open(uploaded_file_yolo)
            st.image(img, caption="📷 Gambar Input", use_container_width=True)
            if st.button("🚀 Jalankan Deteksi", key="detect_btn"):
                with st.spinner("✨ Mendeteksi objek..."):
                    results = yolo_model(img)
                    result_img = results[0].plot()
                    st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        for i, box in enumerate(boxes, 1):
                            label = yolo_model.names[int(box.cls)]
                            conf = box.conf[0]
                            st.write(f"{i}. **{label}** — Confidence: *{conf:.2%}*")
                    else:
                        st.info("Tidak ada objek terdeteksi.")
    except Exception as e:
        st.error(f"❌ Error YOLO: {e}")

# ===================== PROCESS 2 - TF MODEL =====================
with col2:
    st.markdown("""<div class="process-card"><h2>🧠 PROCESS 2</h2><h3>TensorFlow Classification (Pizza / Not Pizza)</h3><div class="process-card-content">""", unsafe_allow_html=True)
    try:
        # ===================== PASTIKAN FOLDER MODEL =====================
        os.makedirs("model", exist_ok=True)

        # ===================== LINK GOOGLE DRIVE =====================
        FILE_ID = "1HbYHvdvdcm0-9bBeJIlrkRrQsY5B0rrZ"
        MODEL_PATH = "model/model_pizza_notpizza.h5"
        DRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

        # ===================== DOWNLOAD MODEL =====================
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024:
            with st.spinner("⬇ Mengunduh model dari Google Drive..."):
                gdown.download(DRIVE_URL, MODEL_PATH, quiet=False, fuzzy=True)
            st.success("✅ Model berhasil diunduh!")

        # ===================== CEK VALIDITAS H5 =====================
        try:
            with h5py.File(MODEL_PATH, 'r') as f:
                st.write("✅ File .h5 valid. Konten:", list(f.keys()))
        except Exception as e:
            st.error(f"❌ File .h5 tidak valid atau korup: {e}")

        # ===================== NAMA KELAS =====================
        class_names = ["Not Pizza", "Pizza"]

        # ===================== CUSTOM LAYER =====================
        class GetItem(Layer):
            def call(self, inputs): 
                return inputs

        @st.cache_resource
        def load_tf_model():
            try:
                return load_model(MODEL_PATH, compile=False, custom_objects={'GetItem': GetItem})
            except Exception:
                return load_model(MODEL_PATH, compile=False)

        with st.spinner("🔄 Memuat model TensorFlow..."):
            model = load_tf_model()
        st.success("✅ Model TensorFlow berhasil dimuat!")

        # ===================== UPLOAD & PREDIKSI =====================
        uploaded_file_tf = st.file_uploader("Upload gambar untuk klasifikasi:", type=["jpg","jpeg","png"], key="tf_pizza")
        if uploaded_file_tf:
            img = Image.open(uploaded_file_tf)
            st.image(img, caption="📷 Gambar Input", use_container_width=True)
            if st.button("🔮 Prediksi Gambar", key="predict_pizza"):
                with st.spinner("✨ Melakukan prediksi..."):
                    img_array = np.array(img.resize((224,224))) / 255.0
                    if len(img_array.shape) == 2: 
                        img_array = np.stack([img_array]*3, axis=-1)
                    elif img_array.shape[-1] == 4: 
                        img_array = img_array[...,:3]
                    img_array = np.expand_dims(img_array, axis=0)

                    predictions = model.predict(img_array, verbose=0)
                    if predictions.shape[-1] == 1:
                        confidence = float(predictions[0][0])
                        predicted_class = "Pizza" if confidence >= 0.5 else "Not Pizza"
                        confidence = confidence if predicted_class=="Pizza" else (1-confidence)
                    else:
                        idx = np.argmax(predictions[0])
                        predicted_class = class_names[idx]
                        confidence = predictions[0][idx]

                    col_a, col_b = st.columns(2)
                    with col_a: st.metric("🎯 Kelas Prediksi", predicted_class)
                    with col_b: st.metric("📊 Confidence", f"{confidence:.2%}")

    except Exception as e:
        st.error(f"❌ Error TensorFlow: {e}")

# ===================== FOOTER =====================
st.markdown("""
<div class="footer">
<p style="margin: 0; font-size: 1.1rem;">
💖 <strong>Dibuat oleh Balqis Isaura</strong> | Powered by Streamlit ✨
</p>
</div>
""", unsafe_allow_html=True)
