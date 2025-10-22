import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
from io import StringIO

# ========================== CONFIG PAGE ==========================
st.set_page_config(
    page_title="Pijjahut - Dini Arifatul Nasywa",
    page_icon="☕",
    layout="wide"
)

# ========================== CUSTOM STYLE ==========================
st.markdown("""
    <style>
        body { background-color: #f9f5f1; }
        .main-title {
            text-align: center;
            font-size: 2.3rem;
            font-weight: 700;
            color: #4b2e05;
            margin-top: 1rem;
        }
        .subtitle {
            text-align: center;
            color: #7a5c2e;
            font-size: 1.1rem;
        }
        .stButton > button {
            background-color: #7b4b22 !important;
            color: white !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.2rem !important;
        }
        .stButton > button:hover {
            background-color: #5c3517 !important;
        }
        .footer {
            text-align: center;
            color: #9b6c3d;
            font-size: 0.9rem;
            margin-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# ========================== HEADER ==========================
st.markdown("<h1 class='main-title'>☕ Pijjahut Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deteksi Objek (Piring & Gelas) dan Klasifikasi Gambar (Pizza vs Not Pizza)</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================== SIDEBAR ==========================
model_choice = st.sidebar.radio("Pilih Model:", ["🔍 YOLO Object Detection", "🍕 ResNet50 Classification"])
st.sidebar.markdown("### ⚙️ Pengaturan")
st.sidebar.info("Gunakan model YOLO untuk mendeteksi objek seperti piring & gelas.\nGunakan model ResNet50 untuk klasifikasi gambar pizza / non-pizza.")

# ========================== YOLO MODEL ==========================
if model_choice == "🔍 YOLO Object Detection":
    st.header("🎯 Deteksi Objek - YOLO")

    @st.cache_resource
    def load_yolo():
        return YOLO('model/DINI ARIFATUL NASYWA_Laporan 4.pt')

    try:
        with st.spinner("Memuat model YOLO..."):
            yolo_model = load_yolo()
        st.success("✅ Model YOLO berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Error memuat model: {e}")
        st.stop()

    uploaded_file = st.file_uploader("Upload gambar untuk deteksi objek", type=["jpg", "jpeg", "png"], key="yolo")

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="📷 Gambar Input", use_column_width=True)

        if st.button("🔍 Deteksi Objek", type="primary"):
            with st.spinner("Mendeteksi objek..."):
                results = yolo_model(image)
                result_img = results[0].plot()

                with col2:
                    st.image(result_img, caption="🎯 Hasil Deteksi", use_column_width=True)

                st.markdown("---")
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

# ========================== TENSORFLOW MODEL ==========================
elif model_choice == "🍕 ResNet50 Classification":
    st.header("🧠 Klasifikasi Gambar - ResNet50")

    @st.cache_resource
    def load_tf_model():
        return tf.keras.models.load_model('model/BISMILLAHDINI2_Laporan2.h5', compile=False)

    try:
        with st.spinner("Memuat model TensorFlow..."):
            tf_model = load_tf_model()
        st.success("✅ Model ResNet50 berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

    uploaded_file = st.file_uploader("Upload gambar untuk klasifikasi", type=["jpg", "jpeg", "png"], key="tf")

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="📷 Gambar Input", use_column_width=True)

        if st.button("🔮 Prediksi Kelas", type="primary"):
            with st.spinner("Melakukan prediksi..."):
                img_array = np.array(image.resize((128, 128)))

                # Handle RGBA -> RGB
                if img_array.shape[-1] == 4:
                    img_array = img_array[:, :, :3]

                img_array = np.expand_dims(img_array, axis=0)
                img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

                predictions = tf_model.predict(img_array, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class]

                # Label mapping
                label_map = {0: "Not Pizza 🚫", 1: "Pizza 🍕"}
                result_label = label_map.get(predicted_class, "Unknown")

                with col2:
                    st.metric("Kelas Prediksi", result_label)
                    st.metric("Confidence", f"{confidence:.2%}")

                with st.expander("📊 Detail Probabilitas"):
                    st.progress(float(predictions[0][0]), text=f"Not Pizza: {predictions[0][0]:.4f}")
                    st.progress(float(predictions[0][1]), text=f"Pizza: {predictions[0][1]:.4f}")

# ========================== FOOTER ==========================
st.markdown("---")
st.markdown("<p class='footer'>© 2025 Dini Arifatul Nasywa | Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
