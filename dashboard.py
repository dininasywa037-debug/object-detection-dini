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
        @import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Dancing+Script&family=Great+Vibes&display=swap'); /* Font menarik tambahan */

        body, .stApp {
            background: linear-gradient(135deg, #fff8dc 0%, #f5f5dc 50%, #ede0c8 100%); /* gradien krem seperti tepung pizza */
            color: #3e2723; /* teks coklat tua untuk kontras */
            font-family: 'Arial', sans-serif;
            overflow-x: hidden; /* mencegah scroll horizontal */
        }

        * {
            color: #3e2723 !important; /* semua teks jadi coklat tua */
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5); /* efek timbul dengan bayangan putih */
        }

        .main-title {
            text-align: center;
            font-size: 8rem; /* diperbesar lagi */
            font-weight: 900;
            font-family: 'Great Vibes', cursive; /* font lebih elegan dan menarik */
            color: #cc0000 !important; /* merah untuk judul agar menonjol */
            text-shadow: 6px 6px 20px rgba(0,0,0,0.5), 0 0 40px rgba(204,0,0,0.8), 0 0 60px rgba(204,0,0,0.4); /* efek timbul dan glow super kuat */
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            animation: bounceIn 3s ease-in-out, pulse 4s infinite, shake 5s infinite;
            position: relative;
        }

        .main-title::before {
            content: '🍕';
            position: absolute;
            left: -50px;
            top: 0;
            font-size: 6rem;
            animation: spin 3s linear infinite;
        }

        .main-title::after {
            content: '🍕';
            position: absolute;
            right: -50px;
            top: 0;
            font-size: 6rem;
            animation: spin 3s linear infinite reverse;
        }

        @keyframes bounceIn {
            0% { transform: scale(0.3); opacity: 0; }
            50% { transform: scale(1.2); }
            70% { transform: scale(0.95); }
            100% { transform: scale(1); opacity: 1; }
        }

        @keyframes pulse {
            0% { text-shadow: 6px 6px 20px rgba(0,0,0,0.5), 0 0 40px rgba(204,0,0,0.8), 0 0 60px rgba(204,0,0,0.4); }
            50% { text-shadow: 6px 6px 20px rgba(0,0,0,0.5), 0 0 50px rgba(204,0,0,1), 0 0 80px rgba(204,0,0,0.6); }
            100% { text-shadow: 6px 6px 20px rgba(0,0,0,0.5), 0 0 40px rgba(204,0,0,0.8), 0 0 60px rgba(204,0,0,0.4); }
        }

        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
            20%, 40%, 60%, 80% { transform: translateX(5px); }
        }

        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .subtitle {
            text-align: center;
            color: #5d4037 !important; /* coklat lebih gelap */
            font-size: 2rem;
            margin-bottom: 2rem;
            font-style: italic;
            text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.7);
            animation: fadeInUp 2.5s ease-in-out;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .section-title {
            font-size: 3rem;
            font-weight: 800;
            color: #ff5722 !important; /* oranye untuk judul section */
            text-shadow: 3px 3px 8px rgba(0,0,0,0.4);
            margin-top: 2rem;
            text-align: center;
            font-family: 'Pacifico', cursive;
            animation: slideInLeft 2s ease-in-out, glow 3s infinite;
        }

        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-60px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes glow {
            0%, 100% { text-shadow: 3px 3px 8px rgba(0,0,0,0.4); }
            50% { text-shadow: 3px 3px 8px rgba(0,0,0,0.4), 0 0 20px rgba(255,87,34,0.8); }
        }

        /* Button style */
        .stButton > button {
            background: linear-gradient(45deg, #ff5722, #e64a19, #d84315) !important; /* gradien oranye-merah lebih kaya */
            color: #fff !important;
            border-radius: 30px !important;
            font-weight: 700 !important;
            padding: 1.5rem 3rem !important;
            box-shadow: 0 8px 25px rgba(0,0,0,0.5);
            transition: all 0.6s ease;
            font-size: 1.3rem;
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
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            transition: left 0.5s;
        }

        .stButton > button:hover::before {
            left: 100%;
        }

        .stButton > button:hover {
            background: linear-gradient(45deg, #e64a19, #bf360c, #8d6e63) !important;
            transform: scale(1.2) rotate(3deg);
            box-shadow: 0 12px 35px rgba(0,0,0,0.7);
        }

        /* Metric cards */
        [data-testid="stMetricValue"], [data-testid="stMetricDelta"], [data-testid="stMetricLabel"] {
            color: #3e2723 !important;
            text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.5);
        }

        /* Expander text */
        .streamlit-expanderHeader, .streamlit-expanderContent {
            color: #3e2723 !important;
        }

        /* Upload box */
        [data-testid="stFileUploader"] label {
            color: #3e2723 !important;
            font-weight: 600;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #5d4037;
            font-size: 1.3rem;
            margin-top: 3rem;
            text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.7);
            animation: fadeIn 3s ease-in-out;
        }

        /* Divider line */
        hr {
            border: 5px solid #d7ccc8;
            border-radius: 20px;
            animation: grow 3s ease-in-out;
            position: relative;
        }

        hr::after {
            content: '🍕🍕🍕';
            position: absolute;
            top: -15px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 1.5rem;
            color: #ff5722;
        }

        @keyframes grow {
            from { width: 0%; }
            to { width: 100%; }
        }

        /* Card style for sections */
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 30px;
            padding: 3rem;
            box-shadow: 0 12px 35px rgba(0,0,0,0.4);
            margin-bottom: 2rem;
            border: 4px solid #d7ccc8;
            transition: transform 0.4s ease, box-shadow 0.4s ease;
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
            transition: all 0.6s;
        }

        .card:hover::before {
            top: -100%;
            left: -100%;
        }

        .card:hover {
            transform: translateY(-15px) scale(1.02);
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        }

        /* Menu recommendation */
        .menu-item {
            background: linear-gradient(45deg, rgba(255, 248, 220, 0.95), rgba(255, 235, 204, 0.95), rgba(255, 222, 173, 0.95));
            border-radius: 25px;
            padding: 2.5rem;
            margin: 0.5rem 0;
            text-align: center;
            font-weight: 600;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            transition: transform 0.5s ease, box-shadow 0.5s ease;
            font-size: 1.2rem;
            position: relative;
        }

        .menu-item:hover {
            transform: translateY(-10px) scale(1.05);
            box-shadow: 0 12px 30px rgba(0,0,0,0.4);
        }

        .menu-item::after {
            content: '🍕';
            position: absolute;
            bottom: 10px;
            right: 10px;
            font-size: 2rem;
            opacity: 0.7;
        }

        /* Tabs style */
        .stTabs [data-baseweb="tab-list"] {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 25px;
            padding: 0.5rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }

        .stTabs [data-baseweb="tab"] {
            color: #3e2723 !important;
            font-weight: 600;
            font-size: 1.3rem;
            transition: all 0.4s ease;
            border-radius: 15px;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(255, 87, 34, 0.3) !important;
            transform: scale(1.05);
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #ff5722 !important;
            color: #fff !important;
            border-radius: 20px;
            transform: scale(1.1);
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        }

        /* Image styling */
        img {
            border-radius: 25px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
        }

        img:hover {
            transform: scale(1.08);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }

        /* Testimonial style */
        .testimonial {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            font-style: italic;
            text-align: center;
            transition: transform 0.3s ease;
        }

        .testimonial:hover {
            transform: translateY(-5px);
        }

        /* Particle effect (simulasi) */
        .particle {
            position: absolute;
            width: 10px;
            height: 10px;
            background: #ff5722;
            border-radius: 50%;
            animation: float 6s infinite ease-in-out;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px); opacity: 0; }
            50% { transform: translateY(-20px); opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

# ========================== HEADER ==========================
st.markdown("<h1 class='main-title'>🍕 Pijjahut 🍕</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Selamat Datang di Restoran Pizza Terbaik! Deteksi Piring & Gelasmu, Klasifikasikan Pizza atau Not Pizza, dan Dapatkan Rekomendasi Menu Spesial! 🔥</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================== HORIZONTAL NAVIGATION (Tabs at Top) ==========================
tabs = st.tabs(["🏠 Beranda", "🔍 Deteksi Objek", "🔮 Klasifikasi Gambar", "🍽️ Menu Rekomendasi", "📞 Kontak Kami", "ℹ️ Tentang Kami"])

# ========================== MAIN CONTENT BASED ON TABS ==========================
with tabs[0]:  # 🏠 Beranda
    st.markdown("<h2 class='section-title'>Selamat Datang di Pijjahut! 🎉</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.5rem;'>Di sini, kami menggabungkan kecanggihan AI dengan cita rasa pizza yang luar biasa! Upload gambar piring atau gelas Anda untuk deteksi objek, atau klasifikasikan apakah itu pizza atau bukan. Berdasarkan hasilnya, kami rekomendasikan menu spesial untuk Anda!</p>
        <p>🍕 <strong>Fitur Utama:</strong></p>
        <ul>
            <li>🔍 Deteksi Piring & Gelas dengan YOLO</li>
            <li>🔮 Klasifikasi Pizza/Not Pizza dengan ResNet50</li>
            <li>🍽️ Rekomendasi Menu Personal</li>
        </ul>
        <p>Jangan ragu untuk eksplorasi! 🚀</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Testimonial Section
    st.markdown("<h3 style='text-align: center; color: #ff5722; font-family: Pacifico, cursive; font-size: 2.5rem;'>Apa Kata Pengguna Kami? 🌟</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='testimonial'>'Pizza di sini luar biasa! AI-nya keren banget!' - User A 🍕</div>", unsafe_allow_html=True)
        st.markdown("<div class='testimonial'>'Rekomendasi menu berdasarkan klasifikasi sangat akurat!' - User B 🔥</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='testimonial'>'Deteksi objeknya cepat dan tepat!' - User C 🔍</div>", unsafe_allow_html=True)
        st.markdown("<div class='testimonial'>'Pengalaman kuliner yang inovatif!' - User D ❤️</div>", unsafe_allow_html=True)
    
    st.image("https://pin.it/6FRyOIyem", caption="Deteksi Piring & Gelas di Restoran Kami", use_container_width=True)

with tabs[1]:  # 🔍 Deteksi Objek
    st.markdown("<h2 class='section-title'>Deteksi Objek di Dapur Kami 🔍</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p>Upload gambar piring atau gelas favorit Anda, dan biarkan AI kami mendeteksinya! Ini seperti memiliki koki pribadi yang memeriksa peralatan makan Anda. 🍽️</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload Gambar Piring/Gelas", type=["jpg", "jpeg", "png"], key="yolo")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Gambar Input Anda", use_container_width=True)

        if st.button("🔍 Deteksi Sekarang!", type="primary", key="detect_obj"):
            try:
                yolo_model = YOLO('model/DINI ARIFATUL NASYWA_Laporan 4.pt')
                results = yolo_model(image)
                result_img = results[0].plot()
                st.image
