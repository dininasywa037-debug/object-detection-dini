import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from io import StringIO

st.set_page_config(
    page_title="Dashboard Model - Dini Arifatul Nasywa", # Diperbarui
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Dashboard Model - Dini Arifatul Nasywa") # Diperbarui
st.markdown("---")

# Sidebar untuk pilih model
model_choice = st.sidebar.radio(
    "Pilih Model:",
    ["PyTorch - YOLO (Balqis)", "TensorFlow - Model Kustom (Dini)"] # Label diperbarui
)

# ==================== MODEL PYTORCH YOLO (Tetap menggunakan model kawan Anda) ====================
if model_choice == "PyTorch - YOLO (Balqis)":
    st.header("🎯 Model PyTorch - YOLO (Balqis Isaura)")
    
    try:
        # Load YOLO model
        @st.cache_resource
        def load_yolo():
            # PATH MODEL KAWAN ANDA
            return YOLO('model/DINI ARIFATUL NASYWA_Laporan 4.pt') 
        
        with st.spinner("Loading YOLO model..."):
            model = load_yolo()
        
        st.success("✅ Model YOLO berhasil dimuat!")
        
        # Info model di sidebar
        with st.sidebar.expander("📊 Info Model"):
            st.text(str(model.info()))
        
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
                        # Mengubah ke format RGB agar Streamlit bisa menampilkan dari numpy array YOLO
                        result_img_rgb = Image.fromarray(result_img[..., ::-1]) 
                        st.image(result_img_rgb, use_column_width=True)
                    
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
        st.info("Pastikan file model YOLO kawan Anda ada di folder 'model/'")

# ==================== MODEL TENSORFLOW (Disesuaikan untuk Dini) ====================
elif model_choice == "TensorFlow - Model Kustom (Dini)":
    st.header("🧠 Model TensorFlow - Model Kustom (Dini Arifatul Nasywa)")
    
    # Resolusi Input yang Disesuaikan untuk Model Kustom (64x64 untuk menghindari error 9216)
    TARGET_SIZE = (64, 64) 
    
    model = None
    
    try:
        # Load TensorFlow model
        @st.cache_resource
        def load_tensorflow():
            # PATH MODEL KUSTOM ANDA
            return tf.keras.models.load_model(
                'model/DINIARIFATUL_Laporan2.h5', 
                compile=False
            )
        
        with st.spinner("Loading TensorFlow model..."):
            model = load_tensorflow()
        
        st.success(f"✅ Model berhasil dimuat! Input disetel ke {TARGET_SIZE} agar sesuai arsitektur.")
        
    except Exception as e:
        st.warning(f"⚠ Gagal load model: {str(e)[:150]}...")
        
        # Coba dengan safe_mode=False
        try:
            st.info("Mencoba metode alternatif...")
            
            @st.cache_resource
            def load_tensorflow_safe():
                import keras
                keras.config.disable_traceback_filtering()
                return tf.keras.models.load_model(
                    'model/DINIARIFATUL_Laporan2.h5',
                    compile=False,
                    safe_mode=False
                )
            
            model = load_tensorflow_safe()
            st.success("✅ Model berhasil dimuat!")
            
        except Exception as e2:
            st.error(f"❌ Model tidak bisa dimuat")
            st.code(str(e2), language="text")
            st.info("""
            *Kemungkinan masalah:*
            - Versi TensorFlow/Keras tidak kompatibel
            - Model perlu disave ulang dengan format terbaru
            - Pastikan file DINIARIFATUL_Laporan2.h5 ada di folder 'model/'.
            """)
    
    # Jika model berhasil di-load
    if model is not None:
        # Info model
        with st.sidebar.expander("📊 Architecture Model"):
            stream = StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            st.text(stream.getvalue())
        
        # Upload gambar
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
                st.subheader(f"📷 Gambar Input (Diresize ke {TARGET_SIZE})")
                
                # Menampilkan gambar input yang sudah di resize
                resized_image_display = image.resize(TARGET_SIZE)
                st.image(resized_image_display, use_column_width=True)
            
            if st.button("🔮 Prediksi", type="primary"):
                with st.spinner("Melakukan prediksi..."):
                    # --- MULAI PREPROCESS KRUSIAL ---
                    # 1. Resize gambar ke ukuran yang disesuaikan
                    img_array = np.array(image.resize(TARGET_SIZE)) 
                    
                    # 2. Convert RGBA to RGB jika perlu
                    if img_array.shape[-1] == 4:
                        img_array = img_array[:, :, :3]
                    
                    # 3. Normalisasi
                    # Model kustom biasanya menggunakan normalisasi 0-1 (dibagi 255.0)
                    img_array = img_array.astype('float32') / 255.0 
                    
                    # 4. Tambahkan dimensi batch
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # --- AKHIR PREPROCESS KRUSIAL ---
                    
                    # Prediksi
                    try:
                        predictions = model.predict(img_array, verbose=0)
                    except Exception as pred_error:
                        st.error(f"❌ Terjadi kesalahan saat prediksi: {pred_error}")
                        st.warning(f"Coba ubah TARGET_SIZE dari {TARGET_SIZE} ke ukuran lain (misalnya 96x96 atau 48x48) jika error dimensi tetap terjadi.")
                        st.stop() # <-- SOLUSI: Mengganti 'return' dengan 'st.stop()'

                    
                    with col2:
                        st.subheader("🎯 Hasil Prediksi")
                        
                        # Asumsi model Anda memiliki output tunggal (misalnya Sigmoid) atau Softmax dengan beberapa kelas
                        if predictions.shape[-1] == 1:
                            # Klasifikasi Biner (Contoh: Pizza vs Bukan Pizza)
                            confidence = predictions[0][0]
                            predicted_class_name = "Class 1" if confidence > 0.5 else "Class 0"
                            st.metric("Kelas Prediksi", predicted_class_name)
                            st.metric("Confidence (Class 1)", f"{confidence:.2%}")
                            st.progress(float(confidence), text=f"Class 1: {confidence:.4f}")
                        else:
                            # Klasifikasi Multi-kelas (Softmax)
                            predicted_class = np.argmax(predictions[0])
                            confidence = predictions[0][predicted_class]
                            
                            st.metric("Kelas Prediksi", f"Class {predicted_class}")
                            st.metric("Confidence", f"{confidence:.2%}")
                            
                            # Tampilkan semua probabilitas
                            with st.expander("📊 Lihat Semua Probabilitas"):
                                for i, prob in enumerate(predictions[0]):
                                    st.progress(float(prob), text=f"Class {i}: {prob:.4f}")

# Footer
st.markdown("---")
st.markdown("📌 Dibuat oleh Dini Arifatul Nasywa | Powered by Streamlit 🚀")
