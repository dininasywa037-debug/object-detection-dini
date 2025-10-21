import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
from io import StringIO

# ✅ Setup halaman Streamlit
st.set_page_config(
    page_title="Dashboard Model - Dini Arifatul Nasywa",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Dashboard Model - Dini Arifatul Nasywa")
st.markdown("---")

# ==================== SIDEBAR ====================
model_choice = st.sidebar.radio(
    "Pilih Model:",
    ["PyTorch - YOLO", "TensorFlow - ResNet50"]
)

# ==================== MODEL YOLO ====================
if model_choice == "PyTorch - YOLO":
    st.header("🎯 Model PyTorch - YOLO")

    try:
        @st.cache_resource
        def load_yolo():
            # Pastikan path model sesuai struktur repo kamu
            return YOLO('DINI ARIFATUL NASYWA_Laporan4.pt')
        
        with st.spinner("Loading YOLO model..."):
            model = load_yolo()
        st.success("✅ Model YOLO berhasil dimuat!")

        # Info model
        with st.sidebar.expander("📊 Info Model"):
            try:
                st.text(str(model.info()))
            except Exception:
                st.write("Model info tidak dapat ditampilkan di Streamlit Cloud.")

        # Upload gambar
        st.markdown("### Upload Gambar untuk Deteksi Objek")
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=['jpg', 'jpeg', 'png'],
            key='yolo'
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)

            if st.button("🔍 Deteksi Objek", type="primary"):
                with st.spinner("Mendeteksi objek..."):
                    results = model(image)
                    result_img = results[0].plot()

                with col2:
                    st.subheader("🎯 Hasil Deteksi")
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
                            label = model.names[int(box.cls)]
                            conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                            st.metric(f"Objek {i+1}", label, f"{conf:.1%}")
                else:
                    st.info("ℹ Tidak ada objek terdeteksi")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Pastikan file model ada di folder 'model/' dan path-nya benar.")

# ==================== MODEL TENSORFLOW ====================
elif model_choice == "TensorFlow - ResNet50":
    st.header("🧠 Model TensorFlow - ResNet50")

    model = None
    try:
        @st.cache_resource
        def load_tensorflow():
            return tf.keras.models.load_model(
                'model/Dini_Arifatul_Nasywa_laporan2.h5',
                compile=False
            )

        with st.spinner("Loading TensorFlow model..."):
            model = load_tensorflow()
        st.success("✅ Model berhasil dimuat!")

    except Exception as e:
        st.warning(f"⚠ Gagal load model: {str(e)[:150]}...")
        try:
            st.info("Mencoba metode alternatif...")
            @st.cache_resource
            def load_tensorflow_safe():
                import keras
                keras.config.disable_traceback_filtering()
                return tf.keras.models.load_model(
                    'model/Dini_Arifatul_Nasywa_laporan2.h5',
                    compile=False,
                    safe_mode=False
                )
            model = load_tensorflow_safe()
            st.success("✅ Model berhasil dimuat!")
        except Exception as e2:
            st.error("❌ Model tidak bisa dimuat")
            st.code(str(e2), language="text")
            st.info("""
            *Kemungkinan masalah:*
            - Versi TensorFlow/Keras tidak kompatibel
            - Model perlu disimpan ulang dengan format terbaru (.h5)
            
            *Solusi sementara:* Gunakan model YOLO yang sudah berfungsi
            """)

    # Jika model berhasil di-load
    if model is not None:
        with st.sidebar.expander("📊 Architecture Model"):
            stream = StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            st.text(stream.getvalue())

        st.markdown("### Upload Gambar untuk Prediksi")
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=['jpg', 'jpeg', 'png'],
            key='tf'
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)

            if st.button("🔮 Prediksi", type="primary"):
                with st.spinner("Melakukan prediksi..."):
                    img_array = np.array(image.resize((224, 224)))

                    if img_array.shape[-1] == 4:
                        img_array = img_array[:, :, :3]

                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

                    predictions = model.predict(img_array, verbose=0)

                with col2:
                    st.subheader("🎯 Hasil Prediksi")
                    predicted_class = int(np.argmax(predictions[0]))
                    confidence = float(predictions[0][predicted_class])

                    st.metric("Kelas Prediksi", f"Class {predicted_class}")
                    st.metric("Confidence", f"{confidence:.2%}")

                    with st.expander("📊 Lihat Semua Probabilitas"):
                        for i, prob in enumerate(predictions[0]):
                            st.progress(float(prob), text=f"Class {i}: {prob:.4f}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("📌 **Dibuat oleh Dini Arifatul Nasywa** | Powered by Streamlit 🚀")
