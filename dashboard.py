import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np

# ========================== CONFIG PAGE ==========================
st.set_page_config(
    page_title="Pijjahut - Dini Arifatul Nasywa",
    page_icon="🍕",
    layout="wide"
)

# ========================== CUSTOM STYLE ==========================
st.markdown("""
    <style>
        body, .stApp {
            background-color: #4e342e; /* coklat tua */
            color: #fff8e1; /* teks krem terang */
        }

        * {
            color: #fff8e1 !important; /* semua teks jadi krem */
            text-shadow: 1px 1px 2px #3e2723; /* efek timbul */
        }

        .main-title {
            text-align: left;
            font-size: 3rem;
            font-weight: 800;
            color: #fffde7 !important;
            text-shadow: 2px 2px 6px #3e2723;
            margin-top: 1rem;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            text-align: left;
            color: #fff8e1 !important;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }

        .section-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #fffde7 !important;
            text-shadow: 1px 1px 3px #3e2723;
            margin-top: 1rem;
        }

        /* Button style */
        .stButton > button {
            background-color: #cc0000 !important;
            color: #fff8e1 !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            padding: 0.6rem 1.2rem !important;
            box-shadow: 0 3px 8px rgba(0,0,0,0.4);
        }

        .stButton > button:hover {
            background-color: #b71c1c !important;
            transform: scale(1.03);
            transition: 0.2s ease-in-out;
        }

        /* Metric cards */
        [data-testid="stMetricValue"], [data-testid="stMetricDelta"], [data-testid="stMetricLabel"] {
            color: #fffde7 !important;
            text-shadow: 1px 1px 3px #3e2723;
        }

        /* Expander text */
        .streamlit-expanderHeader, .streamlit-expanderContent {
            color: #fff8e1 !important;
        }

        /* Upload box */
        [data-testid="stFileUploader"] label {
            color: #fff8e1 !important;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #fff8e1;
            font-size: 0.9rem;
            margin-top: 3rem;
            text-shadow: 1px 1px 3px #3e2723;
        }

        /* Divider line */
        hr {
            border: 1px solid #fff8e155;
        }
    </style>
""", unsafe_allow_html=True)

# ========================== HEADER ==========================
st.markdown("<h1 class='main-title'>Pijjahut</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deteksi Piring & Gelasmu, lalu klasifikasikan yang kamu mau!<br>Pizza atau Not Pizza, dua-duanya tersedia kok 🍕</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================== SECTIONS ==========================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<h2 class='section-title'>Deteksi Objek</h2>", unsafe_allow_html=True)
    st.image("https://pin.it/6FRyOIyem", caption="Deteksi Piring & Gelas", use_container_width=True)
    uploaded_file = st.file_uploader("Upload gambar untuk deteksi objek", type=["jpg", "jpeg", "png"], key="yolo")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Gambar Input", use_container_width=True)

        if st.button("🔍 Deteksi Objek", type="primary", key="detect_obj"):
            try:
                yolo_model = YOLO('model/DINI ARIFATUL NASYWA_Laporan 4.pt')
                results = yolo_model(image)
                result_img = results[0].plot()
                st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)
                st.subheader("📋 Detail Deteksi")
                boxes = results[0].boxes
                if len(boxes) > 0:
                    cols = st.columns(3)
                    for i, box in enumerate(boxes):
                        with cols[i % 3]:
                            st.metric(
                                f"Objek {i+1}",
                                yolo_model.names[int(box.cls)],
                                f"{box.conf[0]:.1%}"
                            )
                else:
                    st.info("ℹ Tidak ada objek terdeteksi.")
            except Exception as e:
                st.error(f"❌ Error memuat model YOLO: {e}")

with col2:
    st.markdown("<h2 class='section-title'>Klasifikasi Gambar</h2>", unsafe_allow_html=True)
    st.image("https://pin.it/5tBvmbECo", caption="Klasifikasi Pizza / Not Pizza", use_container_width=True)
    uploaded_file_tf = st.file_uploader("Upload gambar untuk klasifikasi", type=["jpg", "jpeg", "png"], key="tf")

    if uploaded_file_tf:
        image_tf = Image.open(uploaded_file_tf)
        st.image(image_tf, caption="📷 Gambar Input", use_container_width=True)

        if st.button("🔮 Prediksi Kelas", type="primary", key="predict_tf"):
            try:
                tf_model = tf.keras.models.load_model('model/BISMILLAHDINI2_Laporan2.h5', compile=False)
                img_array = np.array(image_tf.resize((128, 128)))
                if img_array.shape[-1] == 4:
                    img_array = img_array[:, :, :3]
                img_array = np.expand_dims(img_array, axis=0)
                img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

                predictions = tf_model.predict(img_array, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class]

                label_map = {0: "Not Pizza 🚫", 1: "Pizza 🍕"}
                result_label = label_map.get(predicted_class, "Unknown")

                st.metric("Kelas Prediksi", result_label)
                st.metric("Confidence", f"{confidence:.2%}")

                with st.expander("📊 Detail Probabilitas"):
                    st.progress(float(predictions[0][0]), text=f"Not Pizza: {predictions[0][0]:.4f}")
                    st.progress(float(predictions[0][1]), text=f"Pizza: {predictions[0][1]:.4f}")
            except Exception as e:
                st.error(f"❌ Error memuat model ResNet50: {e}")

# ========================== FOOTER ==========================
st.markdown("---")
st.markdown("<p class='footer'>© 2025 Dini Arifatul Nasywa | Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
