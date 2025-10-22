import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import os # Import os untuk pengecekan file model

# ========================== CONFIG PAGE ==========================
st.set_page_config(
    page_title="Pijjahut - Dini Arifatul Nasywa",
    page_icon="🍕",
    layout="wide"
)

# ========================== CUSTOM STYLE (DIRETAIN) ==========================
# (Mempertahankan semua custom CSS Anda yang sudah keren)
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

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
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
        
        .contact-info {
            background: #fff8dc;
            padding: 1.5rem;
            border-radius: 20px;
            border: 2px dashed #ff5722;
            text-align: center;
            font-size: 1.2rem;
            margin-top: 1rem;
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

# ========================== UTILITY FUNCTIONS (Load Models) ==========================
# Caching model agar tidak perlu load ulang setiap kali interaksi
@st.cache_resource
def load_yolo_model(path):
    if not os.path.exists(path):
        st.error(f"File model YOLO tidak ditemukan di: {path}")
        return None
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        return None

@st.cache_resource
def load_classification_model():
    # Ganti dengan path model ResNet50 Anda yang sudah dilatih
    # Jika Anda menggunakan model ResNet50 standar ImageNet, biarkan seperti ini
    try:
        model = tf.keras.applications.ResNet50(weights='imagenet')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model Klasifikasi: {e}. Pastikan Anda memiliki koneksi internet untuk mengunduh weights ImageNet.")
        return None

# ========================== HORIZONTAL NAVIGATION (Tabs at Top) ==========================
tabs = st.tabs(["Beranda 🏠", "Deteksi Objek 🍽️", "Klasifikasi Gambar 🍕", "Menu Rekomendasi 🌟", "Kontak Kami 📞", "Tentang Kami ℹ️"])

# ========================== MAIN CONTENT BASED ON TABS ==========================

# ----------------- BERANDA -----------------
with tabs[0]:
    st.markdown("<h2 class='section-title'>Selamat Datang di Pijjahut</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.4rem;'>Kami menggabungkan kecanggihan **Kecerdasan Buatan (AI)** dengan cita rasa pizza yang luar biasa. Coba fitur **Deteksi Objek** kami untuk mengenali peralatan makan, atau gunakan **Klasifikasi Gambar** untuk mengetahui apakah itu pizza, dan dapatkan **Rekomendasi Menu** personal dari kami!</p>
        <p><strong>Fitur Canggih:</strong></p>
        <ul>
            <li>**Deteksi Cepat:** Mengenali piring dan gelas dengan model **YOLO** yang terlatih.</li>
            <li>**Klasifikasi Cerdas:** Mengidentifikasi gambar sebagai pizza atau bukan pizza menggunakan arsitektur **ResNet50**.</li>
            <li>**Personalisasi:** Rekomendasi menu yang disesuaikan dengan hasil klasifikasi Anda.</li>
        </ul>
        <p>Mari kita mulai petualangan kuliner digital Anda!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Testimonial Section
    st.markdown("<h3 style='text-align: center; color: #ff5722; font-family: Pacifico, cursive; font-size: 2rem;'>Apa Kata Pengguna Kami</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='testimonial'>'Pizza di sini luar biasa. **AI-nya sangat keren**; deteksi piringnya cepat dan tepat!' - Pengguna A, <span style='color:#ff5722;'>Food Blogger</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='testimonial'>'Rekomendasi menu berdasarkan klasifikasi **sangat akurat** dan bikin penasaran.' - Pengguna B, <span style='color:#ff5722;'>Pelanggan Setia</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='testimonial'>'Mengunggah foto dan langsung tahu itu pizza atau bukan. **Pengalaman kuliner yang inovatif**.' - Pengguna C, <span style='color:#ff5722;'>Tech Enthusiast</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='testimonial'>'Desain web yang cantik dan fungsional. Saya suka **estetika Pijjahut**!' - Pengguna D, <span style='color:#ff5722;'>Desainer Grafis</span></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1574071318508-1cdbab80d002?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3wzOTgyNTZ8MHwxfHNlYXJjaHwyMHx8cGl6emElMjByZXN0YXVyYW50fGVufDB8fHx8MTcwNjgzMTc1OHww&ixlib=rb-4.0.3&q=80&w=1080", caption="Suasana Pijjahut: Penggabungan AI dan Cita Rasa", use_container_width=True)


# ----------------- DETEKSI OBJEK -----------------
with tabs[1]:
    st.markdown("<h2 class='section-title'>Deteksi Objek di Meja Makan 🍽️</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p>Kami menggunakan model **YOLO (You Only Look Once)** yang canggih untuk mengidentifikasi **Piring** dan **Gelas**. Coba upload gambar peralatan makan Anda, dan saksikan AI kami bekerja!</p>
    </div>
    """, unsafe_allow_html=True)
    
    yolo_model = load_yolo_model('model/DINI ARIFATUL NASYWA_Laporan 4.pt')
    
    if yolo_model:
        uploaded_file = st.file_uploader("Upload Gambar Piring atau Gelas (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], key="yolo")

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Input Anda", use_container_width=True)

            if st.button("Deteksi Sekarang 🚀", type="primary", key="detect_obj"):
                with st.spinner("⏳ Memproses deteksi objek dengan YOLO..."):
                    try:
                        # Melakukan inferensi
                        results = yolo_model(image)
                        
                        # Mengambil gambar dengan bounding box (sudah dilakukan oleh .plot())
                        # results[0].plot() mengembalikan array numpy (BGR)
                        result_img = results[0].plot() 
                        
                        # Konversi dari BGR ke RGB untuk tampilan Streamlit
                        result_img_rgb = Image.fromarray(result_img[..., ::-1])
                        
                        st.image(result_img_rgb, caption="Hasil Deteksi YOLO", use_container_width=True)
                        st.success("Deteksi berhasil! Objek piring/gelas telah ditandai.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat deteksi: {e}. Pastikan format gambar dan model benar.")
    else:
        st.warning("Model YOLO tidak dapat dimuat. Pastikan file 'model/DINI ARIFATUL NASYWA_Laporan 4.pt' tersedia.")


# ----------------- KLASIFIKASI GAMBAR -----------------
with tabs[2]:
    st.markdown("<h2 class='section-title'>Klasifikasi Gambar Pizza 🍕</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p>Bingung apakah yang Anda lihat adalah pizza? Upload gambarnya! Model klasifikasi berbasis **ResNet50** kami akan memberi tahu Anda. Hasil ini akan menentukan rekomendasi menu spesial.</p>
    </div>
    """, unsafe_allow_html=True)
    
    classification_model = load_classification_model()
    
    if classification_model:
        uploaded_file_class = st.file_uploader("Upload Gambar untuk Klasifikasi (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], key="classify")

        if uploaded_file_class:
            image_pil = Image.open(uploaded_file_class)
            # Preprocessing untuk ResNet50: resize ke (224, 224)
            image_class = image_pil.resize((224, 224))
            st.image(image_class, caption="Gambar Input Anda (diresize ke 224x224)", use_container_width=True)

            if st.button("Klasifikasikan Sekarang 🔍", type="primary", key="classify_btn"):
                with st.spinner("⏳ Mengklasifikasikan gambar dengan ResNet50..."):
                    try:
                        # Ubah ke numpy array dan normalisasi
                        img_array = np.array(image_class)
                        # Preprocessing ResNet50
                        preprocessed_img = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img_array, axis=0))
                        
                        predictions = classification_model.predict(preprocessed_img)
                        # Decode prediksi untuk model ImageNet
                        decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=3)[0] # Ambil top 3
                        
                        st.markdown("### Hasil Analisis Top 3:")
                        
                        is_pizza = False
                        pizza_keywords = ['pizza', 'cheese_pizza', 'hot_dog', 'bagel'] # Tambahkan keyword yang relevan (tergantung dataset ResNet50)
                        
                        # Tampilkan 3 prediksi teratas
                        for i, (imagenet_id, label, confidence) in enumerate(decoded_predictions):
                            if any(keyword in label.lower() for keyword in pizza_keywords):
                                is_pizza = True
                                st.success(f"**#{i+1}: {label.replace('_', ' ').title()}** (Kepercayaan: **{confidence*100:.2f}%**)")
                            else:
                                st.info(f"#{i+1}: {label.replace('_', ' ').title()} (Kepercayaan: {confidence*100:.2f}%)")

                        if is_pizza:
                            final_result = "Pizza"
                            st.session_state['classification'] = 'pizza'
                            st.balloons()
                        else:
                            final_result = "Bukan Pizza (atau objek makanan/masakan lain)"
                            st.session_state['classification'] = 'not_pizza'
                            st.snow()
                        
                        st.markdown(f"---")
                        st.markdown(f"<p style='font-size: 1.8rem; text-align: center; font-weight: bold; color: #cc0000;'>Kesimpulan AI: {final_result}</p>", unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat klasifikasi: {e}")
    else:
        st.warning("Model Klasifikasi (ResNet50) tidak dapat dimuat. Pastikan TensorFlow terinstal dengan benar.")


# ----------------- MENU REKOMENDASI -----------------
with tabs[3]:
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
            <p style='font-size: 1.5rem; text-align: center; color: #d84315; font-weight: bold;'>🎉 Gambar Anda adalah **PIZZA**! 🎉</p>
            <p style='font-size: 1.1rem; text-align: center;'>Karena Anda suka pizza, kami rekomendasikan untuk mencoba varian lain atau pendamping yang pas!</p>
        </div>
        """, unsafe_allow_html=True)
        
        with col_rec1:
            st.markdown("### 🍕 Varian Pizza Wajib Coba")
            for item in menu['pizza_spesial']:
                st.markdown(f"<div class='menu-item'>**{item['nama']}** <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#ff5722; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown("### 🍹 Minuman & Dessert")
            for item in menu['dessert_minuman']:
                st.markdown(f"<div class='menu-item'>**{item['nama']}** <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#ff5722; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
                
    elif st.session_state['classification'] == 'not_pizza':
        st.markdown("""
        <div class='card' style='background: linear-gradient(45deg, #e1f5fe, #b3e5fc); border-color: #0288d1;'>
            <p style='font-size: 1.5rem; text-align: center; color: #01579b; font-weight: bold;'>🥗 Gambar Anda **BUKAN PIZZA**!</p>
            <p style='font-size: 1.1rem; text-align: center;'>Jika Anda mencari alternatif selain pizza, coba menu non-pizza andalan kami. Dijamin tidak kalah lezat!</p>
        </div>
        """, unsafe_allow_html=True)
        
        with col_rec1:
            st.markdown("### 🍝 Pilihan Non-Pizza Terbaik")
            for item in menu['non_pizza_spesial']:
                st.markdown(f"<div class='menu-item'>**{item['nama']}** <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#ff5722; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
                
        with col_rec2:
            st.markdown("### 🍰 Dessert untuk Melengkapi")
            for item in menu['dessert_minuman']:
                st.markdown(f"<div class='menu-item'>**{item['nama']}** <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#ff5722; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
        
    else:
        st.info("💡 **Silakan lakukan Klasifikasi Gambar (Tab ke-3) terlebih dahulu** untuk mendapatkan rekomendasi menu personal yang paling akurat.")


# ----------------- KONTAK KAMI -----------------
with tabs[4]:
    st.markdown("<h2 class='section-title'>Hubungi Kami 📞</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.2rem; text-align: center;'>Ada pertanyaan, masukan, atau ingin memesan langsung? Jangan ragu untuk menghubungi tim Pijjahut.</p>
        <div class='contact-info'>
            <p>📍 **Alamat:** Jl. Digitalisasi No. 101, Kota Streamlit, Kode Pos 404 </p>
            <p>📞 **Telepon:** (021) 123-PIZZA (74992)</p>
            <p>📧 **Email:** **<a href='mailto:pijjahut.ai@gmail.com' style='color: #cc0000 !important; text-decoration: none;'>pijjahut.ai@gmail.com</a>**</p>
            <p>🕒 **Jam Buka:** Setiap Hari, 10:00 - 22:00 WIB</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Lokasi Kami")
    # Contoh peta fiktif
    st.image("https://images.unsplash.com/photo-1549448130-cf220197d0fd?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3wzOTgyNTZ8MHwxfHNlYXJjaHw3fHxpbnN0YWdyYW0lMjBwYWdlfGVufDB8fHx8MTcwNjgzMTU0OXww&ixlib=rb-4.0.3&q=80&w=1080", caption="Ikuti Kami di Media Sosial: @PijjahutAI", use_container_width=True)


# ----------------- TENTANG KAMI -----------------
with tabs[5]:
    st.markdown("<h2 class='section-title'>Tentang Pijjahut ℹ️</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='card'>
        <p style='font-size: 1.4rem;'>Pijjahut didirikan dengan visi untuk membawa teknologi **Kecerdasan Buatan** ke ranah kuliner. Kami percaya bahwa data dan algoritma dapat meningkatkan pengalaman bersantap Anda.</p>
        <p>Proyek ini dikembangkan oleh **Dini Arifatul Nasywa** sebagai bagian dari eksplorasi pada topik:</p>
        <ul>
            <li>**Deteksi Objek (YOLOv8):** Digunakan untuk mengenali peralatan makan dasar, piring dan gelas.</li>
            <li>**Klasifikasi Gambar (ResNet50):** Dimanfaatkan untuk mengidentifikasi produk utama kami: Pizza.</li>
            <li>**Platform:** Dibangun menggunakan **Streamlit** untuk tampilan antarmuka yang interaktif dan *user-friendly*.</li>
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
