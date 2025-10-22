import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import time  # Untuk animasi loading

# ========================== CONFIG PAGE ==========================
st.set_page_config(
    page_title="Pijjahut - Dini Arifatul Nasywa",
    page_icon="🍕",
    layout="wide"
)

# ========================== CUSTOM STYLE ==========================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Dancing+Script&family=Great+Vibes&display=swap');

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

        .main-title {
            text-align: center;
            font-size: 6rem;
            font-weight: 900;
            font-family: 'Great Vibes', cursive;
            color: #cc0000 !important;
            text-shadow: 4px 4px 15px rgba(0,0,0,0.5), 0 0 30px rgba(204,0,0,0.6);
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            animation: bounceIn 2s ease-in-out, pulse 3s infinite;
            position: relative;
        }

        .main-title::before {
            content: '🍕';
            position: absolute;
            left: -40px;
            top: 0;
            font-size: 4rem;
            animation: spin 3s linear infinite;
        }

        .main-title::after {
            content: '🍕';
            position: absolute;
            right: -40px;
            top: 0;
            font-size: 4rem;
            animation: spin 3s linear infinite reverse;
        }

        @keyframes bounceIn {
            0% { transform: scale(0.3); opacity: 0; }
            50% { transform: scale(1.1); }
            70% { transform: scale(0.95); }
            100% { transform: scale(1); opacity: 1; }
        }

        @keyframes pulse {
            0% { text-shadow: 4px 4px 15px rgba(0,0,0,0.5), 0 0 30px rgba(204,0,0,0.6); }
            50% { text-shadow: 4px 4px 15px rgba(0,0,0,0.5), 0 0 40px rgba(204,0,0,0.8); }
            100% { text-shadow: 4px 4px 15px rgba(0,0,0,0.5), 0 0 30px rgba(204,0,0,0.6); }
        }

        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .subtitle {
            text-align: center;
            color: #5d4037 !important;
            font-size: 1.8rem;
            margin-bottom: 2rem;
            font-style: italic;
            text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.7);
            animation: fadeInUp 2s ease-in-out;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
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

        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-40px); }
            to { opacity: 1; transform: translateX(0); }
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

        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            transition: left 0.4s;
        }

        .stButton > button:hover::before {
            left: 100%;
        }

        .stButton > button:hover {
            background: linear-gradient(45deg, #e64a19, #bf360c);
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
        }

        [data-testid="stMetricValue"], [data-testid="stMetricDelta"], [data-testid="stMetricLabel"] {
            color: #3e2723 !important;
            text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.5);
        }

        .streamlit-expanderHeader, .streamlit-expanderContent {
            color: #3e2723 !important;
        }

        [data-testid="stFileUploader"] label {
            color: #3e2723 !important;
            font-weight: 600;
        }

        .footer {
            text-align: center;
            color: #5d4037;
            font-size: 1.2rem;
            margin-top: 3rem;
            text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.7);
            animation: fadeIn 2s ease-in-out;
        }

        hr {
            border: 3px solid #d7ccc8;
            border-radius: 15px;
            animation: grow 2s ease-in-out;
            position: relative;
        }

        hr::after {
            content: '🍕';
            position: absolute;
            top: -10px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 1.2rem;
            color: #ff5722;
        }

        @keyframes grow {
            from { width: 0%; }
            to { width: 100%; }
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

        .card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            transform: rotate(45deg);
            transition: all 0.5s;
        }

        .card:hover::before {
            top: -100%;
            left: -100%;
        }

        .card:hover {
            transform: translateY(-10px) scale(1.01);
            box-shadow: 0 15px 40px rgba(0,0,0,0.4);
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

        .menu-item:hover {
            transform: translateY(-8px) scale(1.03);
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }

        .stTabs [data-baseweb="tab-list"] {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 0.5rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .stTabs [data-baseweb="tab"] {
            color: #3e2723 !important;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            border-radius: 12px;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(255, 87, 34, 0.2) !important;
            transform: scale(1.02);
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #ff5722 !important;
            color: #fff !important;
            border-radius: 15px;
            transform: scale(1.05);
            box-shadow: 0 3px 10px rgba(0,0,0,0.3);
        }

        img {
            border-radius: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        img:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        }

        .testimonial {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            font-style: italic;
            text-align: center;
            transition: transform 0.3s ease;
        }

        .testimonial:hover {
            transform: translateY(-3px);
        }

        .loading {
            text-align: center;
            font-size: 1.5rem;
            color: #ff5722;
            animation: pulse 1s infinite;
        }
    </style>
""", unsafe_allow_html=True)

# ========================== HEADER ==========================
st.markdown("<h1 class='main-title'>Pijjahut</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Selamat datang di restoran pizza terbaik. Deteksi piring dan gelas Anda, klasifikasikan gambar pizza, dan dapatkan rekomendasi menu spesial.</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================== HORIZONTAL NAVIGATION (Tabs at Top) ==========================
tabs = st.tabs(["Beranda", "Deteksi Objek", "Klasifikasi Gambar", "Menu Rekomendasi", "Kontak Kami", "Tentang Kami"])

# ========================== MAIN CONTENT BASED ON TABS ==========================
with tabs[0]:  # Beranda
    st.markdown("<h2 class='section-title'>Selamat Datang di Pijjahut</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.4rem;'>Kami menggabungkan kecanggihan AI dengan cita rasa pizza yang luar biasa. Upload gambar piring atau gelas Anda untuk deteksi objek, atau klasifikasikan apakah itu pizza atau bukan. Berdasarkan hasilnya, kami rekomendasikan menu spesial untuk Anda.</p>
        <p><strong>Fitur Utama:</strong></p>
        <ul>
            <li>Deteksi piring dan gelas dengan YOLO</li>
            <li>Klasifikasi pizza atau bukan pizza dengan ResNet50</li>
            <li>Rekomendasi menu personal</li>
        </ul>
        <p>Jangan ragu untuk menjelajahi fitur-fitur kami.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Testimonial Section
    st.markdown("<h3 style='text-align: center; color: #ff5722; font-family: Pacifico, cursive; font-size: 2rem;'>Apa Kata Pengguna Kami</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='testimonial'>'Pizza di sini luar biasa. AI-nya sangat keren.' - Pengguna A</div>", unsafe_allow_html=True)
        st.markdown("<div class='testimonial'>'Rekomendasi menu berdasarkan klasifikasi sangat akurat.' - Pengguna B</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='testimonial'>'Deteksi objeknya cepat dan tepat.' - Pengguna C</div>", unsafe_allow_html=True)
        st.markdown("<div class='testimonial'>'Pengalaman kuliner yang inovatif.' - Pengguna D</div>", unsafe_allow_html=True)
    
    st.image("https://pin.it/6FRyOIyem", caption="Deteksi piring dan gelas di restoran kami", use_container_width=True)

with tabs[1]:  # Deteksi Objek
    st.markdown("<h2 class='section-title'>Deteksi Objek di Dapur Kami</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p>Upload gambar piring atau gelas favorit Anda, dan biarkan AI kami mendeteksinya. Ini seperti memiliki koki pribadi yang memeriksa peralatan makan Anda.</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Gambar Piring atau Gelas", type=["jpg", "jpeg", "png"], key="yolo")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Input Anda", use_container_width=True)

        if st.button("Deteksi Sekarang", type="primary", key="detect_obj"):
            with st.spinner("Memproses deteksi..."):
                time.sleep(1)  # Simulasi loading
                try:
                    yolo_model = YOLO('model/DINI ARIFATUL NASYWA_Laporan 4.pt')
                    results = yolo_model(image)
                    result_img = results[0].plot()
                    st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
                    st.success("Deteksi berhasil! Lihat hasil di atas.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

with tabs[2]:  # Klasifikasi Gambar
    st.markdown("<h2 class='section-title'>Klasifikasi Gambar Pizza</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p>Upload gambar apa saja, dan AI kami akan mengklasifikasikan apakah itu pizza atau bukan. Hasilnya akan digunakan untuk rekomendasi menu.</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file_class = st.file_uploader("Upload Gambar untuk Klasifikasi", type=["jpg", "jpeg", "png"], key="classify")

    if uploaded_file_class:
        image_class = Image.open(uploaded_file_class).resize((224, 224))
        st.image(image_class, caption="Gambar Input Anda", use_container_width=True)

        if st.button("Klasifikasikan Sekarang", type="primary", key="classify_btn"):
            with st.spinner("Mengklasifikasikan gambar..."):
                time.sleep(1)  # Simulasi loading
                try:
                    # Load pre-trained ResNet50 model (asumsi model sudah dilatih untuk pizza/not pizza)
                    model = tf.keras.applications.ResNet50(weights='imagenet')  # Ganti dengan model Anda jika ada
                    img_array = np.array(image_class) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    predictions = model.predict(img_array)
                    decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0]
                    label = decoded_predictions[0][1]
                    confidence = decoded_predictions[0][2]
                    
                    if 'pizza' in label.lower():
                        result = "Pizza"
                        st.session_state['classification'] = 'pizza'
                    else:
                        result = "Bukan Pizza"
                        st.session_state['classification'] = 'not_pizza'
                    
                    st.success(f"Hasil Klasifikasi: {result} (Kepercayaan: {confidence:.2f})")
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

with tabs[3]:  # Menu Rekomendasi
    st.markdown("<h2 class='section-title'>Rekomendasi Menu</h2>", unsafe_allow_html=True)
    if 'classification' in st.session_state:
        if st.session_state['classification'] == 'pizza':
            st.markdown("""
            <div class='card'>
                <p style='font-size: 1.4rem;'>Karena gambar Anda terdeteksi sebagai pizza, kami rekomendasikan menu berikut:</p>
            </div>
            """, unsafe_allow_html=True
