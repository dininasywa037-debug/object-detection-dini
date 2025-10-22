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
        @import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Dancing+Script&family=Great+Vibes&display=swap'); /* Font tambahan untuk lebih elegan */

        body, .stApp {
            background: linear-gradient(135deg, #fff8dc 0%, #f5f5dc 50%, #ede0c8 100%); /* gradien krem seperti tepung pizza */
            color: #3e2723; /* teks coklat tua untuk kontras */
            font-family: 'Arial', sans-serif;
            overflow-x: hidden; /* cegah scroll horizontal */
        }

        * {
            color: #3e2723 !important; /* semua teks jadi coklat tua */
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5); /* efek timbul dengan bayangan putih */
        }

        .main-title {
            text-align: center;
            font-size: 8rem; /* diperbesar lagi untuk lebih dramatis */
            font-weight: 900;
            font-family: 'Great Vibes', cursive; /* font lebih elegan dan mewah */
            color: #cc0000 !important; /* merah untuk judul agar menonjol */
            text-shadow: 6px 6px 20px rgba(0,0,0,0.5), 0 0 40px rgba(204,0,0,0.8), 0 0 60px rgba(204,0,0,0.4); /* efek timbul, glow, dan halo lebih kuat */
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            animation: bounceIn 3s ease-in-out, pulse 4s infinite, shake 5s infinite;
            position: relative;
        }

        .main-title::before {
            content: '🍕';
            position: absolute;
            left: -50px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 4rem;
            animation: spin 3s linear infinite;
        }

        .main-title::after {
            content: '🍕';
            position: absolute;
            right: -50px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 4rem;
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
            animation: slideInLeft 2s ease-in-out;
        }

        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-60px); }
            to { opacity: 1; transform: translateX(0); }
        }

        /* Button style */
        .stButton > button {
            background: linear-gradient(45deg, #ff5722, #e64a19, #d84315) !important; /* gradien 3 warna untuk lebih keren */
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
            transition: left 0.6s;
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
            box-shadow: 0 12px 40px rgba(0,0,0,0.4);
            margin-bottom: 2rem;
            border: 4px solid #d7ccc8;
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            position: relative;
        }

        .card::before {
            content: '';
            position: absolute;
            top: -10px;
            left: -10px;
            right: -10px;
            bottom: -10px;
            background: linear-gradient(45deg, #ff5722, #e64a19);
            border-radius: 35px;
            z-index: -1;
            opacity: 0;
            transition: opacity 0.4s ease;
        }

        .card:hover::before {
            opacity: 0.3;
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

        .menu-item::after {
            content: '🍕';
            position: absolute;
            bottom: 10px;
            right: 10px;
            font-size: 2rem;
            opacity: 0.7;
        }

        .menu-item:hover {
            transform: translateY(-10px) scale(1.05) rotate(1deg);
            box-shadow: 0 12px 30px rgba(0,0,0,0.4);
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
            border-radius: 20px;
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
            box-shadow: 0 4px 15px rgba(255,87,34,0.5);
        }

        /* Image styling */
        img {
            border-radius: 25px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            transition: transform 0.4s ease, filter 0.4s ease;
        }

        img:hover {
            transform: scale(1.08) rotate(2deg);
            filter: brightness(1.1);
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

        /* Sidebar style */
        .sidebar .sidebar-content {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            border: 2px solid #d7ccc8;
        }

        /* Carousel effect for images (simple) */
        .carousel {
            display: flex;
            overflow: hidden;
            border-radius: 20px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }

        .carousel img {
            width: 100%;
            transition: transform 0.5s ease;
        }

        /* Particle effect simulation with CSS */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }

        .particle {
            position: absolute;
            background: rgba(255, 87, 34, 0.1);
            border-radius: 50%;
            animation: float 10s infinite linear;
        }

        @keyframes float {
            from { transform: translateY(100vh) rotate(0deg); }
            to { transform: translateY(-100px) rotate(360deg); }
        }
    </style>
""", unsafe_allow_html=True)

# ========================== PARTICLE EFFECT (Simulasi) ==========================
st.markdown("""
    <div class="particles">
        <div class="particle" style="left: 10%; animation-delay: 0s; width: 20px; height: 20px;"></div>
        <div class="particle" style="left: 20%; animation-delay: 1s; width: 15px; height: 15px;"></div>
        <div class="particle" style="left: 30%; animation-delay: 2s; width: 25px; height: 25px;"></div>
        <div class="particle" style="left: 40%; animation-delay: 3s; width: 18px; height: 18px;"></div>
        <div class="particle" style="left: 50%; animation-delay: 4s; width: 22px; height: 22px;"></div>
        <div class="particle" style="left: 60%; animation-delay: 5s; width: 16px; height: 16px;"></div>
        <div class="particle" style="left: 70%; animation-delay: 6s; width: 24px; height: 24px;"></div>
        <div class="particle" style="left: 80%; animation-delay: 7s; width: 19px; height: 19px;"></div>
        <div class="particle" style="left: 90%; animation-delay: 8s; width: 21px; height: 21px;"></div>
    </div>
""", unsafe_allow_html=True)

# ========================== HEADER ==========================
st.markdown("<h1 class='main-title'>Pijjahut</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Selamat Datang di Restoran Pizza Terbaik! Deteksi Piring & Gelasmu, Klasifikasikan Pizza atau Not Pizza, dan Dapatkan Rekomendasi Menu Spesial! 🔥</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================== HORIZONTAL NAVIGATION (Tabs at Top) ==========================
tabs = st.tabs(["🏠 Beranda", "🔍 Deteksi Objek", "🔮 Klasifikasi Gambar", "🍽️ Menu Rekomendasi", "📞 Kontak Kami"])

# ========================== MAIN CONTENT BASED ON TABS ==========================
with tabs[0]:  # 🏠 Beranda
    # Sidebar di Beranda untuk Tentang Kami
    with st.sidebar:
        st.markdown("## ℹ️ Tentang Kami")
        st.markdown("""
        <div class='sidebar-content'>
            <p>Pijjahut adalah restoran pizza inovatif yang menggunakan AI untuk mendeteksi objek dan mengklasifikasikan makanan. Kami percaya bahwa teknologi dapat meningkatkan pengalaman kuliner Anda!</p>
            <p>🍕 <strong>Misi Kami:</strong> Menyediakan pizza berkualitas tinggi dengan sentuhan kecanggihan AI.</p>
            <p>❤️ <strong>Tim Kami:</strong> Dibuat oleh Dini Arifatul Nasywa dengan cinta dan dedikasi.</p>
            <p>🚀 Bergabunglah dengan kami untuk pengalaman kuliner yang unik dan tak terlupakan!</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h2 class='section-title'>Selamat Datang di Pijjahut! 🎉</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.5rem;'>Di sini, kami menggabungkan kecanggihan AI dengan cita rasa pizza yang luar biasa! Upload
