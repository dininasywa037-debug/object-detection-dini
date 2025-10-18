import os
import streamlit as st
from PIL import Image
import numpy as np

# ===========================
# Konfigurasi YOLO
# ===========================
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.makedirs("/tmp/Ultralytics", exist_ok=True)

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
    ["PyTorch - YOLO", "TensorFlow - ResNet50"]
)

# ===========================
# Fungsi Load Model
# ===========================
@st.cache_resource
def load_yolo_model():
    try:
        from ultralytics import YOLO
        return YOLO("model/DINI ARIFATUL_NASYWA_Laporan 4.pt")
    except Exception as e:
        st.warning(f"⚠ YOLO model tidak ditemukan atau gagal load: {e}")
        return None

@st.cache_resource
def load_tf_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model("model/Dini_Arifatul_Nasywa_laporan2.h5", compile=False)
        return model
    except Exception as e:
        st.warning(f"⚠ TensorFlow model tidak ditemukan atau gagal load: {str(e)[:150]}...")
        return None

# ===========================
# MODEL PYTORCH YOLO
# ===========================
if model_choice == "PyTorch - YOLO":
    st.header("🎯 Model PyTorch - YOLO")
    model = load_yolo_model()
    
    if model is not None:
        uploaded_file = st.file_uploader("Unggah gambar untuk deteksi YOLO:", type=['jpg', 'jpeg', 'png'], key='yolo')

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_array = np.array(image)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)

            if st.button("🔍 Deteksi Objek", key='yolo_button'):
                with st.spinner("Mendeteksi objek..."):
                    results = model(image_array)
                    result_img = results[0].plot()

                    with col2:
                        st.subheader("🎯 Hasil Deteksi")
                        st.image(result_img, use_column_width=True)

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
    else:
        st.info("⚠ YOLO model belum tersedia.")

# ===========================
# MODEL TENSORFLOW
# ===========================
elif model_choice == "TensorFlow - ResNet50":
    st.header("🧠 Model TensorFlow - ResNet50")
    model = load_tf_model()

    if model is not None:
        uploaded_file = st.file_uploader("Unggah gambar untuk prediksi:", type=['jpg', 'jpeg', 'png'], key='tf')

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            # Konversi gambar aman
            if len(img_array.shape) == 2:  # grayscale
                img_array = np.stack([img_array]*3, axis=-1)
            elif img_array.shape[-1] == 4:  # RGBA
                img_array = img_array[:, :, :3]

            img_array = np.expand_dims(np.array(Image.fromarray(img_array).resize((224, 224))), axis=0)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)

            if st.button("🔮 Prediksi", key='tf_button'):
                with st.spinner("Melakukan prediksi..."):
                    import tensorflow as tf
                    from tensorflow.keras.applications.resnet50 import preprocess_input
                    x = preprocess_input(img_array.astype(np.float32))
                    preds = model.predict(x, verbose=0)

                    predicted_class = np.argmax(preds[0])
                    confidence = preds[0][predicted_class]

                    with col2:
                        st.subheader("🎯 Hasil Prediksi")
                        st.metric("Kelas Prediksi", f"Class {predicted_class}")
                        st.metric("Confidence", f"{confidence:.2%}")

                        with st.expander("📊 Semua Probabilitas"):
                            for i, prob in enumerate(preds[0]):
                                st.progress(float(prob), text=f"Class {i}: {prob:.4f}")
    else:
        st.info("⚠ TensorFlow model belum tersedia.")

# Footer
st.markdown("---")
st.markdown("📌 Dibuat oleh Balqis Isaura | Powered by Streamlit 🚀")
