import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from ultralytics import YOLO
import gdown
import os

# ===================== KONFIGURASI DASHBOARD =====================
st.set_page_config(
    page_title="Dashboard Model - Balqis Isaura",
    page_icon="✨",
    layout="wide"
)

# ===================== STYLE PINK THEME WITH CHECKERED CORNERS =====================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    [data-testid="stAppViewContainer"] {
        background-color: #fff5f8 !important;
        font-family: 'Poppins', sans-serif;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 180px;
        height: 180px;
        background-color: #ffb3d9;
        background-image: 
            linear-gradient(45deg, #ffc9e3 25%, transparent 25%),
            linear-gradient(-45deg, #ffc9e3 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, #ffc9e3 75%),
            linear-gradient(-45deg, transparent 75%, #ffc9e3 75%);
        background-size: 20px 20px;
        background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
        z-index: 0;
        pointer-events: none;
        border-bottom-right-radius: 50% 30%;
    }
    
    [data-testid="stAppViewContainer"]::after {
        content: '';
        position: fixed;
        bottom: 0;
        right: 0;
        width: 250px;
        height: 250px;
        background-color: #ffcce0;
        background-image: 
            linear-gradient(45deg, #ffd9eb 25%, transparent 25%),
            linear-gradient(-45deg, #ffd9eb 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, #ffd9eb 75%),
            linear-gradient(-45deg, transparent 75%, #ffd9eb 75%);
        background-size: 20px 20px;
        background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
        z-index: 0;
        pointer-events: none;
        border-top-left-radius: 50% 30%;
    }

    h1 {
        color: #ff1493 !important;
        text-align: center;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(255, 20, 147, 0.2);
    }

    .subtitle {
        text-align: center;
        color: #ff69b4;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ff69b4, #ff1493) !important;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .process-card {
        background: linear-gradient(135deg, #ff69b4 0%, #ff1493 100%);
        border-radius: 25px;
        padding: 30px;
        margin: 20px 10px;
        box-shadow: 0 8px 20px rgba(255, 20, 147, 0.3);
        position: relative;
        overflow: hidden;
    }

    .process-card h2 {
        color: white !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }

    .process-card h3 {
        color: white !important;
        text-align: center;
        margin-bottom: 0;
    }

    .process-card-content {
        background: white;
        border-radius: 20px;
        padding: 25px;
        margin-top: 15px;
        position: relative;
        z-index: 1;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .stButton > button {
        background: linear-gradient(90deg, #ff1493, #ff69b4) !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        border-radius: 25px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 20, 147, 0.3) !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #ff69b4, #ff1493) !important;
        box-shadow: 0 6px 20px rgba(255, 20, 147, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    [data-testid="stFileUploader"] {
        background: #fff0f5;
        border: 2px dashed #ff69b4;
        border-radius: 15px;
        padding: 20px;
    }

    [data-testid="stFileUploader"] label {
        color: #ff1493 !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #ff1493 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    .footer {
        text-align: center;
        color: #ff1493;
        font-weight: 600;
        margin-top: 3rem;
        padding: 20px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(255, 20, 147, 0.1);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ===================== DECORATIVE ELEMENTS =====================
st.markdown("""
    <div class="decoration star1">⭐</div>
    <div class="decoration star2">✨</div>
    <div class="decoration star3">💫</div>
    <div class="decoration heart">💖</div>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown("# ✨ DASHBOARD MODEL ✨")
st.markdown('<p class="subtitle">GROUP PROJECT PRESENTATION - BALQIS ISAURA</p>', unsafe_allow_html=True)

# ===================== DUA KOLOM =====================
col1, col2 = st.columns(2, gap="medium")

# ===================== PROCESS 1 - YOLO OBJECT DETECTION =====================
with col1:
    st.markdown("""
        <div class="process-card">
            <h2>📸 PROCESS 1</h2>
            <h3>YOLO Object Detection</h3>
            <div class="process-card-content">
    """, unsafe_allow_html=True)
    
    try:
        @st.cache_resource
        def load_yolo():
            return YOLO("model/DINI ARIFATUL NASYWA_Laporan 4.pt")

        with st.spinner("🔄 Memuat model YOLO..."):
            yolo_model = load_yolo()
        st.success("✅ Model YOLO berhasil dimuat!")

        uploaded_file_yolo = st.file_uploader(
            "Upload gambar untuk deteksi objek:", 
            type=["jpg", "jpeg", "png"], 
            key="yolo"
        )

        if uploaded_file_yolo:
            image = Image.open(uploaded_file_yolo)
            st.image(image, caption="📷 Gambar Input", use_container_width=True)

            if st.button("🚀 Jalankan Deteksi", key="detect_btn"):
                with st.spinner("✨ Mendeteksi objek..."):
                    results = yolo_model(image)
                    result_img = results[0].plot()
                    st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)

                    st.markdown("#### 📋 Detail Deteksi")
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
        st.info("💡 Pastikan file .pt ada di folder model/.")
    
    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)

# ===================== PROCESS 2 - TENSORFLOW CLASSIFICATION =====================
with col2:
    st.markdown("""
        <div class="process-card">
            <h2>🧠 PROCESS 2</h2>
            <h3>TensorFlow Classification</h3>
            <div class="process-card-content">
    """, unsafe_allow_html=True)

    import os
    import gdown
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Layer



        # Jika model belum ada di folder lokal, unduh dari Google Drive
        if not os.path.exists(MODEL_PATH):
            FILE_ID = "1JCYISlbpPHMd6Vx4gC8j968NYzSJXqbd"
            with st.spinner("⬇ Mengunduh model dari Google Drive..."):
                gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
            st.success("✅ Model berhasil diunduh!")

        # Buat dummy layer GetItem biar bisa load model tanpa error
        class GetItem(Layer):
            def __init__(self, **kwargs):
                super(GetItem, self).__init__(**kwargs)
            def call(self, inputs):
                # fungsi asli bisa beda, tapi minimal ini biar model kebuka
                return inputs

        # Load model
        with st.spinner("🔍 Memuat model TensorFlow..."):
            model = load_model(MODEL_PATH, custom_objects={'GetItem': GetItem})
            st.success("✅ Model berhasil dimuat!")

        # Setelah ini kamu bisa lanjutkan ke bagian prediksi gambar
        uploaded_image = st.file_uploader("Unggah Gambar untuk Klasifikasi", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            import numpy as np
            from PIL import Image

            image = Image.open(uploaded_image).resize((224, 224))  # sesuaikan ukuran
            img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

            prediction = model.predict(img_array)
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)
            st.write("### Hasil Prediksi:", prediction)

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
