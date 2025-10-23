import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from io import BytesIO

# ========================== CONFIG PAGE ==========================
st.set_page_config(
    page_title="Pijjahut - Dini Arifatul Nasywa",
    page_icon="🍕",
    layout="wide"
)

# ========================== CUSTOM STYLE ==========================
st.markdown("""
    <style>
        @import url('https://fonts.com/css2?family=Pacifico&family=Dancing+Script&family=Great+Vibes&display=swap');

        /* === Efek Transisi Fade-In Global untuk Tampilan Tab yang Lebih Halus === */
        .stApp {
            animation: fadeInAnimation ease 0.4s; 
            animation-iteration-count: 1;
            animation-fill-mode: forwards;
        }

        @keyframes fadeInAnimation {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        /* ====================================================================== */

        /* === PERUBAHAN UNTUK MEMUSATKAN TABS NAVIGASI === */
        div[data-testid="stDecoration"] + div {
            display: flex;
            justify-content: center;
        }

        div[data-testid="stDecoration"] + div > div:first-child {
            width: auto;
        }
        /* ==================================================== */
        
        /* === Tampilan Tabs Navigasi (Dipercantik) === */
        div[role="tablist"] {
            padding: 0.5rem 1rem;
            margin-bottom: 2rem; 
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.6); 
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            border: 1px solid #ffccbc;
        }

        .stTabs [data-testid="stTab"] {
            border-radius: 10px;
            padding: 0.5rem 1rem;
            margin: 0 5px;
            color: #3e2723 !important;
            font-weight: 600 !important;
            transition: all 0.3s ease;
            background-color: transparent;
            border: none;
        }

        .stTabs [data-testid="stTab"]:hover {
            background-color: #ffccbc !important; 
            transform: translateY(-2px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .stTabs [data-testid="stTab"][aria-selected="true"] {
            color: #cc0000 !important; 
            border-bottom: 3px solid #cc0000; 
            font-weight: 700 !important;
        }
        /* ================================================== */


        body, .stApp {
            background: linear-gradient(135deg, #fff8dc 0%, #f5f5dc 50%, #ede0c8 100%);
            color: #3e2723;
            font-family: 'Arial', sans-serif;
            overflow-x: hidden;
        }

        * {
            color: #3e2723 !important;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5); 
        }

        /* === JUDUL UTAMA === */
        .main-title {
            text-align: center;
            font-size: clamp(4rem, 10vw, 8rem);
            font-weight: 900;
            font-family: 'Great Vibes', cursive;
            color: #cc0000 !important;
            
            /* Efek shadow/glow */
            text-shadow: 
                -3px -3px 0px rgba(255, 255, 255, 0.8), 
                3px 3px 0px #8b0000, 
                0 0 15px rgba(255, 0, 0, 0.9), 
                0 0 30px rgba(204, 0, 0, 0.5); 
                
            margin-top: 2rem; 
            margin-bottom: 0.5rem;
            animation: bounceIn 2s ease-in-out, pulse 3s infinite;
            position: relative;
            width: 100%;
            display: block; 
            letter-spacing: 12px;
        }
        /* ============================================ */

        .subtitle {
            text-align: center;
            color: #5d4037 !important;
            font-size: 1.8rem;
            margin-bottom: 2rem;
            font-style: italic;
            text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.7);
            animation: fadeInUp 2s ease-in-out;
        }
        .section-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #ff5722 !important;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
            margin-top: 2rem;
            text-align: center;
            font-family: 'Pacifico', cursive;
            animation: slideInLeft 1.5s ease-in-out;
        }
        .stButton > button {
            background: linear-gradient(45deg, #ff5722, #e64a19);
            color: #fff !important;
            border-radius: 25px !important;
            font-weight: 600 !important;
            padding: 1rem 2rem !important;
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            transition: all 0.4s ease;
            font-size: 1.1rem;
            border: none;
            position: relative;
            overflow: hidden;
        }
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 25px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin-bottom: 2rem;
            border: 3px solid #d7ccc8;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .menu-item {
            background: linear-gradient(45deg, rgba(255, 248, 220, 0.95), rgba(255, 235, 204, 0.95));
            border-radius: 20px;
            padding: 2rem;
            margin: 0.5rem 0;
            text-align: center;
            font-weight: 600;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            font-size: 1.1rem;
            position: relative;
        }
        
        /* CUSTOM CSS FILE UPLOADER */
        .stFileUploader {
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .stFileUploader > div:first-child > div:first-child {
            background-color: rgba(255, 255, 255, 0.9);
            border: 3px dashed #ff5722; 
            border-radius: 25px; 
            padding: 2.5rem 1.5rem; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
            transition: all 0.3s ease;
        }
        .stFileUploader > div:first-child > div:first-child:hover {
            border: 3px dashed #cc0000;
            background-color: #fff8e1;
        }
        .stFileUploader button {
            background: linear-gradient(45deg, #cc0000, #a00000); 
            color: #fff !important;
            border: none;
            border-radius: 15px !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: bold;
            box-shadow: 0 3px 10px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }
        .stFileUploader .css-1by9afl p {
            font-size: 1.1rem;
            font-weight: 600;
            color: #d84315 !important;
            text-shadow: none;
        }

        /* CUSTOM CSS PESAN PERINGATAN REKOMENDASI */
        .recommendation-alert {
            padding: 1.5rem;
            border: 3px solid #ffcc00; 
            border-radius: 15px;
            background: linear-gradient(45deg, #fff3e0, #ffecb3); 
            text-align: center;
            margin-top: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            animation: fadeIn 1s ease-out;
        }

        .recommendation-alert p {
            margin: 0;
            color: #e65100 !important; 
            font-size: 1.25rem;
            font-weight: 600;
            text-shadow: none;
        }
        
        .recommendation-alert .icon {
            font-size: 1.8rem;
            margin-right: 10px;
            color: #cc0000 !important; 
        }
        
        .footer {
            text-align: center;
            font-size: 0.9rem;
            margin-top: 1rem;
            color: #795548 !important;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

# ========================== HEADER ==========================
st.markdown("<h1 class='main-title'>Pijjahut</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Selamat datang di restoran pizza terbaik. Deteksi piring dan gelas Anda, klasifikasikan gambar pizza, dan dapatkan rekomendasi menu spesial.</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================== INITIALIZE SESSION STATE ==========================
if 'classification' not in st.session_state:
    st.session_state['classification'] = 'none'
if 'detection_result_img' not in st.session_state:
    st.session_state['detection_result_img'] = None
if 'classification_final_result' not in st.session_state:
    st.session_state['classification_final_result'] = None
if 'classification_image_input' not in st.session_state:
    st.session_state['classification_image_input'] = None
if 'last_yolo_uploader' not in st.session_state:
    st.session_state['last_yolo_uploader'] = None
if 'last_classify_uploader' not in st.session_state:
    st.session_state['last_classify_uploader'] = None

# >>> SESSION STATE UNTUK TESTIMONI <<<
if 'testimonials' not in st.session_state:
    st.session_state['testimonials'] = [
        {'author': 'Balqis, Food Blogger', 'quote': 'Pijjahut luar biasa. AI-nya sangat keren, deteksi piringnya cepat dan tepat!'},
        {'author': 'Oja, Tech Enthusiast', 'quote': 'Mengunggah foto dan langsung tahu itu pizza atau bukan. Pengalaman kuliner yang inovatif!'},
        {'author': 'Syira, Pelanggan Setia', 'quote': 'Rekomendasi menu berdasarkan klasifikasi sangat akurat dan bikin penasaran.'},
        {'author': 'Marlin, Desainer Grafis', 'quote': 'Desain web yang cantik dan fungsional. Saya suka estetika Pijjahut!'}
    ]
# =========================================================

# ========================== UTILITY FUNCTIONS (Load Models) ==========================
@st.cache_resource
def load_yolo_model(path):
    # PATH MODEL DETEKSI ANDA
    YOLO_MODEL_PATH = 'model/DINI ARIFATUL NASYWA_Laporan 4.pt'
    
    # Cek keberadaan file sebelum memuat
    if not os.path.exists(YOLO_MODEL_PATH):
        st.error(f"❌ File model YOLO TIDAK DITEMUKAN di: '{YOLO_MODEL_PATH}'.")
        st.error("Pastikan file DINI ARIFATUL NASYWA_Laporan 4.pt ada di dalam folder 'model'.")
        return None
        
    try:
        model = YOLO(YOLO_MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLO dari '{YOLO_MODEL_PATH}'. Error: {e}")
        return None

@st.cache_resource
def load_classification_model():
    # PATH MODEL KLASIFIKASI KUSTOM 
    MODEL_PATH = 'model/BISMILLAHDINI_Laporan2.h5' 
    
    # Cek keberadaan file sebelum memuat
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ File model klasifikasi TIDAK DITEMUKAN di: '{MODEL_PATH}'.")
        st.error("Pastikan file BISMILLAHDINI_Laporan2.h5 ada di dalam folder 'model'.")
        return None

    try:
        # Mencoba memuat model Keras
        model = tf.keras.models.load_model(MODEL_PATH) 
        return model
    except Exception as e:
        # Pesan error spesifik untuk membantu diagnosa format/integritas file
        st.error(f"Gagal memuat model Klasifikasi kustom dari '{MODEL_PATH}'.")
        st.error(f"Detail Error: {e}")
        st.warning("Penyebab error 'file signature not found' biasanya karena: 1) File korup/tidak selesai di-download, atau 2) File BUKAN dalam format .h5 Keras/TensorFlow yang valid.")
        return None

# ========================== KONTROL STATE SAAT BERPINDAH TAB ==========================
def clear_inactive_results(current_tab_index):
    # Tab Deteksi Objek (Index 1)
    if current_tab_index != 1 and st.session_state.get('detection_result_img') is not None:
        st.session_state['detection_result_img'] = None

    # Tab Klasifikasi Gambar (Index 2)
    if current_tab_index != 2:
        if st.session_state.get('classification_final_result') is not None:
            st.session_state['classification_final_result'] = None
        if st.session_state.get('classification_image_input') is not None:
            st.session_state['classification_image_input'] = None


# ========================== HORIZONTAL NAVIGATION (Tabs at Top) ==========================
tabs = st.tabs(["Beranda 🏠", "Deteksi Objek 🍽️", "Klasifikasi Gambar 🍕", "Menu Rekomendasi 🌟", "Kontak Kami 📞", "Tentang Kami ℹ️"])

# ========================== MAIN CONTENT BASED ON TABS ==========================

# ----------------- BERANDA -----------------
with tabs[0]:
    clear_inactive_results(0)
    st.markdown("<h2 class='section-title'>Jelajahi Inovasi AI Kuliner</h2>", unsafe_allow_html=True)
    
    # KARTU SAMBUTAN
    st.markdown(f"""
    <div class='card' style='background: linear-gradient(180deg, #ffeedd 0%, #fffaf0 100%); border-color: #ff9800;'>
        <p style='font-size: 1.5rem; text-align: center; color: #3e2723;'>
            Kami menyajikan pengalaman kuliner masa depan dengan integrasi <span style='font-weight: bold; color: #cc0000;'>Kecerdasan Buatan (AI)</span>.
            Temukan sajian terbaik, mulai dari deteksi peralatan makan hingga rekomendasi menu personal.
        </p>
        <p style='font-size: 1.1rem; text-align: center; font-style: italic; color: #5d4037;'>
            Satu gigitan, satu algoritma yang sempurna.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # >>> KODE BARU UNTUK FORMULIR TESTIMONI (Di Atas) <<<
    st.markdown("<h2 class='section-title' style='margin-top: 3rem;'>Berikan Testimoni Anda ❤️</h2>", unsafe_allow_html=True)
    
    # Menggunakan form Streamlit agar input dipertahankan saat tombol ditekan
    with st.form(key='testimonial_form', clear_on_submit=True):
        col_form1, col_form2 = st.columns([0.4, 0.6])
        
        with col_form1:
            input_name = st.text_input("Nama Anda dan Profesi (Contoh: Budi, Chef)", max_chars=50)
            submit_button = st.form_submit_button(label='Kirim Testimoni ❤️', type="primary")

        with col_form2:
            input_quote = st.text_area("Testimoni Anda tentang Pijjahut", height=50, max_chars=200)

        if submit_button:
            if input_name and input_quote:
                new_testimonial = {
                    'author': input_name,
                    'quote': input_quote
                }
                # Tambahkan testimoni baru ke session state (di awal list agar langsung terlihat)
                st.session_state['testimonials'].insert(0, new_testimonial)
                st.success("Testimoni Anda berhasil dikirim dan akan muncul di bawah!")
            else:
                st.warning("Mohon lengkapi Nama/Profesi dan Testimoni.")
    
    # Bagian Tampilan Testimoni (Diletakkan di Bawah dalam 4 Kolom)
    st.markdown("<h2 class='section-title' style='margin-top: 3rem;'>Apa Kata Pengguna Kami</h2>", unsafe_allow_html=True)
    
    # Membuat 4 kolom untuk menampung testimoni
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    cols_t = [col_t1, col_t2, col_t3, col_t4] # Daftar kolom
    
    testimonials_list = st.session_state['testimonials']
    
    # Mengiterasi dan menempatkan setiap testimoni ke salah satu dari 4 kolom secara bergantian
    for i, item in enumerate(testimonials_list):
        col = cols_t[i % 4] # Mengganti kolom (0, 1, 2, 3, 0, 1, ...)
        
        with col:
            st.markdown(f"""
            <div class='menu-item' style='min-height: 180px; text-align: left; font-style: italic; font-size: 0.95rem; border: 2px solid #ff5722;'>
                "{item['quote']}"
                <p style='margin-top: 10px; font-weight: bold; font-style: normal; font-size: 0.9rem; color: #a00000;'>
                    - {item['author']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    # >>> AKHIR KODE TESTIMONI BARU <<<


    st.markdown("---")

    # Judul Fitur Canggih Kami
    st.markdown("<h2 class='section-title' style='margin-top: 3rem;'>Fitur Canggih Kami</h2>", unsafe_allow_html=True)
    
    # Bagian Fitur Canggih (3 Kolom)
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        st.markdown("""
        <div class='menu-item'>
            <p style='font-size: 2rem;'>🍽️</p>
            <span style='font-weight: bold; color: #ff5722;'>Deteksi Objek Meja</span>
            <p style='font-size: 0.95rem; margin-top: 0.5rem;'>Mengenali piring dan gelas di meja Anda secara instan menggunakan model <span style='font-weight: bold;'>YOLO</span>.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_feat2:
        st.markdown("""
        <div class='menu-item'>
            <p style='font-size: 2rem;'>✨</p>
            <span style='font-weight: bold; color: #ff5722;'>Klasifikasi Pizza Akurat</span>
            <p style='font-size: 0.95rem; margin-top: 0.5rem;'>Menganalisis gambar untuk mengidentifikasi apakah itu pizza atau bukan dengan <span style='font-weight: bold;'>Model Kustom Keras</span>.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_feat3:
        st.markdown("""
        <div class='menu-item'>
            <p style='font-size: 2rem;'>🎁</p>
            <span style='font-weight: bold; color: #ff5722;'>Rekomendasi Personal</span>
            <p style='font-size: 0.95rem; margin-top: 0.5rem;'>Dapatkan saran menu spesial yang disesuaikan berdasarkan hasil klasifikasi Anda.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True) 
    

# ----------------- DETEKSI OBJEK (Layout VERTICAL) -----------------
with tabs[1]:
    clear_inactive_results(1)
    st.markdown("<h2 class='section-title'>Deteksi Objek di Meja Makan 🍽️</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p>Kami menggunakan model <span style='font-weight: bold;'>YOLO (You Only Look Once)</span> yang canggih untuk mengidentifikasi <span style='font-weight: bold;'>Piring</span> dan <span style='font-weight: bold;'>Gelas</span>. Coba upload gambar peralatan makan Anda, dan saksikan AI kami bekerja!</p>
    </div>
    """, unsafe_allow_html=True)
    
    yolo_model = load_yolo_model('model/DINI ARIFATUL NASYWA_Laporan 4.pt')
    
    if yolo_model:
        
        # --- Bagian Input (FULL WIDTH) ---
        uploaded_file_deteksi = st.file_uploader("Upload Gambar Piring atau Gelas (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], key="yolo_uploader")
        
        # Logika reset hasil jika file baru diupload atau dihapus
        if st.session_state.get('last_yolo_uploader') != uploaded_file_deteksi:
            st.session_state['detection_result_img'] = None
            st.session_state['last_yolo_uploader'] = uploaded_file_deteksi # Update tracker

        if uploaded_file_deteksi:
            image = Image.open(uploaded_file_deteksi)
            st.image(image, caption="Gambar Input Anda", use_container_width=True)

            if st.button("Deteksi Sekarang 🚀", type="primary", key="detect_obj"):
                with st.spinner("⏳ Memproses deteksi objek dengan YOLO..."):
                    try:
                        # Model inference
                        results = yolo_model(image)
                        result_img = results[0].plot()  
                        # Ubah dari BGR (YOLO default) ke RGB (PIL/Streamlit default)
                        result_img_rgb = Image.fromarray(result_img[..., ::-1])
                        
                        st.session_state['detection_result_img'] = result_img_rgb
                        st.success("Deteksi berhasil! Objek piring/gelas telah ditandai.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat deteksi: {e}. Pastikan format gambar dan model benar.")

        st.markdown("---") # Garis pemisah antara input dan output
        
        # --- Bagian Output (Di Bawah Input - FULL WIDTH) ---
        # BARIS INI TELAH DIKOREKSI GAYA DAN INDENTASINYA
        st.markdown("<h2 class='section-title'>Hasil Deteksi</h2>", unsafe_allow_html=True)
        if st.session_state.get('detection_result_img') is not None:
            st.image(st.session_state['detection_result_img'], caption="Hasil Deteksi YOLO", use_container_width=True)
        else:
            # SPASI KOSONG
            st.markdown("<div style='height: 300px;'></div>", unsafe_allow_html=True) 
            
    else:
        st.warning("Model Deteksi YOLO tidak dapat dimuat. Silakan periksa pesan error di atas.")


# ----------------- KLASIFIKASI GAMBAR (Layout VERTICAL) -----------------
with tabs[2]:
    clear_inactive_results(2)
    st.markdown("<h2 class='section-title'>Klasifikasi Gambar Pizza 🍕</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p>Bingung apakah yang Anda lihat adalah pizza? Upload gambarnya! Model klasifikasi berbasis <span style='font-weight: bold;'>Model Kustom (BISMILLAHDINI_Laporan2.h5)</span> akan memberi tahu Anda. Hasil ini akan menentukan rekomendasi menu spesial.</p>
    </div>
    """, unsafe_allow_html=True)
    
    classification_model = load_classification_model()
    
    if classification_model:
        
        # --- Bagian Input (FULL WIDTH) ---
        uploaded_file_class = st.file_uploader("Upload Gambar untuk Klasifikasi (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], key="classify_uploader")

        # Logika reset hasil jika file baru diupload atau dihapus
        if st.session_state.get('last_classify_uploader') != uploaded_file_class:
            st.session_state['classification_final_result'] = None
            st.session_state['classification_image_input'] = None
            st.session_state['last_classify_uploader'] = uploaded_file_class # Update tracker

        if uploaded_file_class:
            image_pil = Image.open(uploaded_file_class)
            
            # PENYESUAIAN UKURAN KE 128x128 
            image_class_resized = image_pil.resize((128, 128)) 
            
            st.session_state['classification_image_input'] = image_class_resized
            
            # Perbarui caption untuk mencerminkan ukuran 128x128
            st.image(st.session_state['classification_image_input'], caption="Gambar Input Anda (diresize ke 128x128 agar sesuai model)", use_container_width=True)

            if st.button("Klasifikasikan Sekarang 🔍", type="primary", key="classify_btn"):
                with st.spinner("⏳ Mengklasifikasikan gambar dengan Model Kustom Anda..."):
                    try:
                        # 1. Konversi dan Pra-pemrosesan 
                        img_array = np.array(image_class_resized, dtype=np.float32)
                        
                        # Handle Grayscale/Alpha channel
                        if img_array.ndim == 2:
                            img_array = np.stack((img_array,)*3, axis=-1)
                        if img_array.shape[2] == 4:
                            img_array = img_array[:,:,:3] 
                        
                        # Normalisasi (Asumsi model dilatih dengan normalisasi 0-1)
                        img_array /= 255.0
                        
                        # Tambahkan dimensi batch 
                        final_img_input = np.expand_dims(img_array, axis=0)

                        # 2. Prediksi Menggunakan Model Kustom
                        predictions = classification_model.predict(final_img_input)
                        
                        # Logika prediksi
                        if predictions.shape[1] == 1:
                            pizza_probability = predictions[0][0] 
                        else:
                            pizza_probability = predictions[0][1] # Asumsi index 1 adalah kelas Pizza
                        
                        # 3. Logika Klasifikasi (THRESHOLD 0.4)
                        THRESHOLD = 0.4 
                        
                        if pizza_probability > THRESHOLD:
                            final_result = "Pizza"
                            st.session_state['classification'] = 'pizza'
                            st.balloons()
                            st.success(f"Probabilitas Klasifikasi: **{pizza_probability:.2f}** (Threshold > {THRESHOLD})")
                        else:
                            final_result = "Bukan Pizza"
                            st.session_state['classification'] = 'not_pizza'
                            st.snow()
                            st.info(f"Probabilitas Klasifikasi: **{pizza_probability:.2f}** (Threshold < {THRESHOLD})")
                        
                        st.session_state['classification_final_result'] = final_result
                        
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat klasifikasi: {e}")
                        st.warning(f"Error Klasifikasi: {e}")


        st.markdown("---") # Garis pemisah antara input dan output
        
        # --- Bagian Output (Di Bawah Input - FULL WIDTH) ---
        if st.session_state.get('classification_final_result') is not None:
            final_result = st.session_state['classification_final_result']
            
            st.markdown("### Hasil Klasifikasi AI")
            
            if final_result == "Pizza":
                st.success(f"🎉 Selamat! Objek ini terklasifikasi sebagai {final_result}.")
            else:
                st.info(f"😕 Objek ini terklasifikasi sebagai {final_result} (mungkin makanan atau masakan lain).")

            st.markdown(f"<p style='font-size: 1.8rem; text-align: center; font-weight: bold; color: #cc0000;'>Kesimpulan AI: {final_result}</p>", unsafe_allow_html=True)
        else:
            # SPASI KOSONG PENGGANTI PLACEHOLDER TEKS
            st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)


    else:
        # Pesan ini akan muncul jika model gagal dimuat
        st.warning("Model Klasifikasi kustom tidak dapat dimuat. Silakan periksa pesan error di atas untuk detail diagnosis.")


# ----------------- MENU REKOMENDASI -----------------
with tabs[3]:
    clear_inactive_results(3)
    st.markdown("<h2 class='section-title'>Rekomendasi Menu Spesial 🌟</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Rekomendasi ini didasarkan pada hasil klasifikasi gambar Anda di tab sebelumnya. Mari kita lihat apa yang cocok untuk Anda!</p>", unsafe_allow_html=True)
    
    menu = {
        'pizza_spesial': [
            {'nama': 'Pizza Margherita Klasik', 'deskripsi': 'Saus tomat otentik, Mozzarella segar, dan daun Basil. Kesempurnaan Italia.', 'harga': 'Rp 85.000'},
            {'nama': 'Pizza Pepperoni AI', 'deskripsi': 'Daging pepperoni premium dan keju yang dideteksi AI, dijamin terbaik.', 'harga': 'Rp 95.000'}
        ],
        'non_pizza_spesial': [
            {'nama': 'Spaghetti Bolognese Pijjahut', 'deskripsi': 'Spaghetti al dente dengan saus daging rahasia yang kaya rasa.', 'harga': 'Rp 65.000'},
            {'nama': 'Caesar Salad Segar', 'deskripsi': 'Salad sehat dengan ayam panggang, crouton, dan dressing Caesar creamy.', 'harga': 'Rp 55.000'}
        ],
        'dessert_minuman': [
            {'nama': 'Lava Cake Cokelat Panas', 'deskripsi': 'Dessert wajib untuk menutup hidangan Anda.', 'harga': 'Rp 40.000'},
            {'nama': 'Es Teh Lemon Segar', 'deskripsi': 'Pendingin yang sempurna setelah makan pizza pedas.', 'harga': 'Rp 20.000'}
        ]
    }
    
    col_rec1, col_rec2 = st.columns(2)
    
    if st.session_state['classification'] == 'pizza':
        st.markdown("""
        <div class='card' style='background: linear-gradient(45deg, #ffe0b2, #ffcc80); border-color: #ff9800;'>
            <p style='font-size: 1.5rem; text-align: center; color: #d84315; font-weight: bold;'>🎉 Gambar Anda adalah <span style='font-weight: bold;'>PIZZA</span>! 🎉</p>
            <p style='font-size: 1.1rem; text-align: center;'>Karena Anda suka pizza, kami rekomendasikan untuk mencoba varian lain atau pendamping yang pas!</p>
        </div>
        """, unsafe_allow_html=True)
        
        with col_rec1:
            st.markdown("### 🍕 Varian Pizza Wajib Coba")
            for item in menu['pizza_spesial']:
                st.markdown(f"<div class='menu-item'><span style='font-weight: bold;'>{item['nama']}</span> <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#ff5722; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown("### 🍹 Minuman & Dessert")
            for item in menu['dessert_minuman']:
                st.markdown(f"<div class='menu-item'><span style='font-weight: bold;'>{item['nama']}</span> <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#ff5722; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
        
    elif st.session_state['classification'] == 'not_pizza':
        st.markdown("""
        <div class='card' style='background: linear-gradient(45deg, #e1f5fe, #b3e5fc); border-color: #0288d1;'>
            <p style='font-size: 1.5rem; text-align: center; color: #01579b; font-weight: bold;'>🥗 Gambar Anda <span style='font-weight: bold;'>BUKAN PIZZA</span>!</p>
            <p style='font-size: 1.1rem; text-align: center;'>Jika Anda mencari alternatif selain pizza, coba menu non-pizza andalan kami. Dijamin tidak kalah lezat!</p>
        </div>
        """, unsafe_allow_html=True)
        
        with col_rec1:
            st.markdown("### 🍝 Pilihan Non-Pizza Terbaik")
            for item in menu['non_pizza_spesial']:
                st.markdown(f"<div class='menu-item'><span style='font-weight: bold;'>{item['nama']}</span> <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#ff5722; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown("### 🍰 Dessert untuk Melengkapi")
            for item in menu['dessert_minuman']:
                st.markdown(f"<div class='menu-item'><span style='font-weight: bold;'>{item['nama']}</span> <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#ff5722; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class='recommendation-alert'>
            <p>
                <span class='icon'>🍕</span> 
                <span style='font-weight: bold;'>Silakan lakukan Klasifikasi Gambar (Tab ke-3) terlebih dahulu</span> untuk mendapatkan rekomendasi menu personal yang paling akurat.<br>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ----------------- KONTAK KAMI -----------------
with tabs[4]:
    clear_inactive_results(4)
    st.markdown("<h2 class='section-title'>Hubungi Kami 📞</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.2rem; text-align: center;'>Ada pertanyaan, masukan, atau ingin memesan langsung? Jangan ragu untuk menghubungi tim Pijjahut.</p>
        <div class='contact-info'>
            <p><span style='font-weight: bold;'>📍 Alamat:</span> Jl. Digitalisasi No. 101, Kota Streamlit, Kode Pos 404 </p>
            <p><span style='font-weight: bold;'>📞 Telepon:</span> (021) 123-PIZZA (74992)</p>
            <p><span style='font-weight: bold;'>📧 Email:</span> <a href='mailto:pijjahut.ai@gmail.com' style='color: #cc0000 !important; text-decoration: none;'>pijjahut.ai@gmail.com</a></p>
            <p><span style='font-weight: bold;'>🕒 Jam Buka:</span> Setiap Hari, 10:00 - 22:00 WIB</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
# ----------------- TENTANG KAMI -----------------
with tabs[5]:
    clear_inactive_results(5)
    st.markdown("<h2 class='section-title'>Tentang Pijjahut ℹ️</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='card'>
        <p style='font-size: 1.4rem;'>Pijjahut didirikan dengan visi untuk membawa teknologi <span style='font-weight: bold;'>Kecerdasan Buatan</span> ke ranah kuliner. Kami percaya bahwa data dan algoritma dapat meningkatkan pengalaman bersantap Anda.</p>
        <p>Proyek ini dikembangkan oleh <span style='font-weight: bold;'>Dini Arifatul Nasywa</span> sebagai bagian dari eksplorasi pada topik:</p>
        <ul>
            <li><span style='font-weight: bold;'>Deteksi Objek (YOLOv8):</span> Digunakan untuk mengenali peralatan makan dasar, piring dan gelas.</li>
            <li><span style='font-weight: bold;'>Klasifikasi Gambar (Model Kustom Keras):</span> Dimanfaatkan untuk mengidentifikasi produk utama kami: Pizza.</li>
            <li><span style='font-weight: bold;'>Platform:</span> Dibangun menggunakan <span style='font-weight: bold;'>Streamlit</span> untuk tampilan antarmuka yang interaktif dan <span style='font-weight: bold;'>user-friendly</span>.</li>
            <li><span style='font-weight: bold;'>Nilai Threshold Klasifikasi:</span> Ditetapkan sebesar <span style='font-weight: bold;'>0.4</span>.</li>
        </ul>
        <p>Terima kasih telah menjadi bagian dari perjalanan inovatif ini!</p>
        <div style='text-align: center; margin-top: 2rem;'>
            <p style='font-style: italic; color: #cc0000;'>#AIxKuliner #Pijjahut #DiniArifatulNasywa</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ========================== FOOTER ==========================
st.markdown("---")
st.markdown("<p class='footer'>© 2024 Pijjahut. Dibuat dengan 🍕 dan ❤️ oleh Dini Arifatul Nasywa.</p>", unsafe_allow_html=True)
