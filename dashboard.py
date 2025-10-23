import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import os

# ========================== CONFIG PAGE ==========================
st.set_page_config(
    page_title="Pijjahut - Dini Arifatul Nasywa",
    page_icon="🍕",
    layout="wide"
)

# ========================== CUSTOM STYLE (MODIFIKASI DAN PERBAIKAN) ==========================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Dancing+Script&family=Great+Vibes&display=swap');

        /* Background dan Font Utama */
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

        /* PERBAIKAN: MAIN TITLE agar tidak tertutup */
        .main-title {
            text-align: center;
            font-size: 8rem; /* Ukuran yang lebih wajar */
            font-weight: 900;
            font-family: 'Great Vibes', cursive;
            color: #cc0000 !important;
            text-shadow: 4px 4px 15px rgba(0,0,0,0.5), 0 0 30px rgba(204,0,0,0.6);
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            animation: bounceIn 2s ease-in-out, pulse 3s infinite;
            position: relative;
            line-height: 1.2; /* Mengurangi jarak antar baris */
        }
        
        @media (max-width: 600px) {
            .main-title {
                font-size: 5rem; /* Ukuran yang lebih kecil untuk mobile */
            }
        }

        /* PERBAIKAN POSISI EMOJI PIZZA */
        .main-title::before {
            content: '🍕';
            position: absolute;
            /* Posisikan di kiri atas judul */
            left: 50%; 
            transform: translateX(-150%) translateY(10%); /* Geser ke kiri dari tengah */
            top: 10%; 
            font-size: 4rem;
            animation: spin 3s linear infinite;
        }

        .main-title::after {
            content: '🍕';
            position: absolute;
            /* Posisikan di kanan atas judul */
            right: 50%; 
            transform: translateX(150%) translateY(10%); /* Geser ke kanan dari tengah */
            top: 10%; 
            font-size: 4rem;
            animation: spin 3s linear infinite reverse;
        }

        /* Animasi */
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
            font-size: 2.8rem; 
            font-weight: 700;
            color: #ff5722 !important;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
            margin-top: 2rem;
            margin-bottom: 1.5rem; 
            text-align: center;
            font-family: 'Pacifico', cursive;
            animation: slideInLeft 1.5s ease-in-out;
        }

        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-40px); }
            to { opacity: 1; transform: translateX(0); }
        }

        /* Button Styling */
        .stButton > button {
            background: linear-gradient(45deg, #ff5722, #e64a19);
            color: #fff !important;
            border-radius: 30px !important; /* Diperbesar */
            font-weight: 600 !important;
            padding: 1rem 2.5rem !important; /* Diperbesar */
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            transition: all 0.4s ease;
            font-size: 1.2rem; /* Diperbesar */
            border: 2px solid #fff; /* Border putih kontras */
            position: relative;
            overflow: hidden;
        }

        .stButton > button:hover {
            background: linear-gradient(45deg, #e64a19, #bf360c);
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
        }
        
        /* Garis Pemisah (HR) */
        hr {
            border: 3px solid #d7ccc8;
            border-radius: 15px;
            animation: grow 2s ease-in-out;
            position: relative;
            margin: 2rem 0; 
        }
        
        /* Card Styling */
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 25px;
            padding: 2.5rem; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin-bottom: 2rem;
            border: 3px solid #ffcc80; /* Warna border lebih cerah */
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .card:hover {
            transform: translateY(-5px) scale(1.01);
            box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        }

        /* Menu Item Styling (Tab Rekomendasi) */
        .menu-item {
            background: linear-gradient(45deg, rgba(255, 248, 220, 0.95), rgba(255, 235, 204, 0.95));
            border-radius: 20px;
            padding: 1.5rem;
            margin: 0.8rem 0; 
            text-align: center;
            font-weight: 600;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            font-size: 1.1rem;
            border: 2px solid #ff5722; /* Border tegas */
        }

        .menu-item:hover {
            transform: translateY(-8px) scale(1.03);
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            background: linear-gradient(45deg, rgba(255, 240, 200, 0.95), rgba(255, 220, 180, 0.95));
        }

        /* Testimonial Styling (Dibuat lebih menonjol) */
        .testimonial {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            font-style: italic;
            text-align: center;
            transition: transform 0.3s ease;
            border-left: 7px solid #ff5722; /* Garis vertikal menonjol */
            border-bottom: 2px solid #ffcc80;
        }

        .testimonial:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }
        
        /* Contact Info Styling */
        .contact-info {
            background: #fff8dc;
            padding: 1.5rem;
            border-radius: 20px;
            border: 2px dashed #ff5722;
            text-align: center;
            font-size: 1.2rem;
            margin-top: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ========================== HEADER ==========================
st.markdown("<h1 class='main-title'>Pijjahut</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Selamat datang di restoran pizza digital kami: **AI meets Culinary Excellence!**</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================== INITIALIZE SESSION STATE ==========================
if 'classification' not in st.session_state:
    st.session_state['classification'] = 'none'

# ========================== UTILITY FUNCTIONS (Load Models) ==========================
@st.cache_resource
def load_yolo_model(path):
    if not os.path.exists(path):
        return None
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        return None

@st.cache_resource
def load_classification_model():
    try:
        model = tf.keras.applications.ResNet50(weights='imagenet')
        return model
    except Exception as e:
        return None

# ========================== HORIZONTAL NAVIGATION (Tabs at Top) ==========================
tabs = st.tabs(["Beranda ✨", "Deteksi Alat Makan 🍽️", "Klasifikasi Makanan 📸", "Menu Rekomendasi 🎁", "Kontak Kami 💌", "Tentang Proyek 💡"])

# ========================== MAIN CONTENT BASED ON TABS ==========================

# ----------------- BERANDA (Fitur Unggulan Kami Dihapus) -----------------
with tabs[0]:
    st.markdown("<h2 class='section-title'>Selamat Datang di Pijjahut: Rasa dan Teknologi!</h2>", unsafe_allow_html=True)
    
    # KONTEN CARD UTAMA (Lebih Ringkas & Fokus)
    st.markdown(f"""
    <div class='card'>
        <p style='font-size: 1.4rem; text-align: center; font-style: italic;'>
            "Mengubah setiap santapan menjadi pengalaman digital yang cerdas."
        </p>
        <br>
        <p style='font-size: 1.2rem;'>
            Kami menghadirkan dimensi baru dalam pengalaman kuliner Anda. Pijjahut menggabungkan <span style='font-weight: bold; color: #cc0000;'>Kecerdasan Buatan (AI)</span> dengan hidangan Italia klasik. Mulai dari **Deteksi Alat Makan** menggunakan YOLO hingga **Klasifikasi Makanan** menggunakan ResNet50, semuanya untuk memastikan Anda mendapat **Rekomendasi Menu Personal** terbaik!
        </p>
        <p style='font-size: 1.2rem;'>
            Ayo, jelajahi tab di atas dan rasakan pengalaman bersantap masa depan!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---") # Visual Separator
    
    # Testimonial Section
    st.markdown("<h3 style='text-align: center; color: #ff5722; font-family: Pacifico, cursive; font-size: 2rem;'>Apa Kata Pengguna Kami 💬</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='testimonial'>'Pizza' di sini luar biasa. <span style='font-weight: bold; color: #cc0000;'>AI-nya sangat keren</span>, deteksi piringnya cepat dan tepat! Saya suka alur kerjanya.' - Balqis, <span style='color:#ff5722;'>Food Blogger</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='testimonial'>'Rekomendasi menu berdasarkan klasifikasi <span style='font-weight: bold; color: #cc0000;'>sangat akurat</span> dan bikin penasaran. Saya coba menu non-pizza dan itu *worth it*.' - Syira, <span style='color:#ff5722;'>Pelanggan Setia</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='testimonial'>'Mengunggah foto dan langsung tahu itu pizza atau bukan. <span style='font-weight: bold; color: #cc0000;'>Pengalaman kuliner yang inovatif</span>. Waktu tunggunya juga cepat.' - Oja, <span style='color:#ff5722;'>Tech Enthusiast</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='testimonial'>'Desain web yang cantik dan fungsional. Saya suka <span style='font-weight: bold; color: #cc0000;'>estetika Pijjahut</span>! Warna dan font-nya sangat menyelerakan.' - Marlin, <span style='color:#ff5722;'>Desainer Grafis</span></div>", unsafe_allow_html=True)
    

# ----------------- DETEKSI OBJEK -----------------
with tabs[1]:
    st.markdown("<h2 class='section-title'>Deteksi Peralatan Makan dengan YOLO 🍽️</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.2rem; text-align: center;'>
            <span style='font-weight: bold; color: #ff5722;'>Fungsi Utama:</span> Identifikasi <span style='font-weight: bold;'>Piring (plate)</span> dan <span style='font-weight: bold;'>Gelas (glass)</span>.
        </p>
        <p>Model <span style='font-weight: bold;'>YOLO (You Only Look Once)</span> yang kami latih memastikan meja Anda siap untuk hidangan berikutnya!</p>
    </div>
    """, unsafe_allow_html=True)
    
    yolo_model = load_yolo_model('model/DINI ARIFATUL NASYWA_Laporan 4.pt')
    
    if yolo_model:
        uploaded_file = st.file_uploader("Upload Gambar Piring atau Gelas (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], key="yolo")

        if uploaded_file:
            st.divider() 
            col_input, col_output = st.columns(2)
            image = Image.open(uploaded_file)
            
            with col_input:
                st.markdown("### Gambar Asli 🖼️")
                st.image(image, caption="Gambar Input Anda", use_container_width=True)

            if st.button("Deteksi Sekarang 🚀", type="primary", key="detect_obj"):
                with st.spinner("⏳ Memproses deteksi objek dengan YOLO..."):
                    try:
                        start_time = time.time()
                        results = yolo_model(image)
                        end_time = time.time()
                        
                        result_img = results[0].plot()
                        result_img_rgb = Image.fromarray(result_img[..., ::-1])
                        
                        detection_count = len(results[0].boxes)
                        
                        with col_output:
                            st.markdown("### Hasil Deteksi ✨")
                            st.image(result_img_rgb, caption="Hasil Deteksi YOLO", use_container_width=True)
                            
                        # Metrik di bawah
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.metric(label="Total Objek Terdeteksi", value=f"{detection_count} 🍽️")
                        with col_info2:
                            st.metric(label="Waktu Pemrosesan", value=f"{end_time - start_time:.3f} s")

                        st.success(f"Deteksi berhasil! Ditemukan {detection_count} objek piring/gelas.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat deteksi: {e}. Pastikan format gambar dan model benar.")
            else:
                with col_output:
                    st.info("Tekan tombol 'Deteksi Sekarang' untuk melihat hasil!")
    else:
        st.warning("Model YOLO tidak dapat dimuat. Pastikan file 'model/DINI ARIFATUL NASYWA_Laporan 4.pt' tersedia di lokasi yang benar.")


# ----------------- KLASIFIKASI GAMBAR -----------------
with tabs[2]:
    st.markdown("<h2 class='section-title'>Klasifikasi Gambar: Pizza atau Bukan? 🧐</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.2rem; text-align: center;'>
            <span style='font-weight: bold; color: #ff5722;'>FUNGSI UTAMA!</span> Hasil klasifikasi ini akan membuka <span style='font-weight: bold;'>Rekomendasi Menu Spesial</span> di tab berikutnya.
        </p>
        <p>Upload gambar makanan Anda. Model <span style='font-weight: bold;'>ResNet50</span> akan memutuskan: **Pizza** atau **Bukan Pizza**.</p>
    </div>
    """, unsafe_allow_html=True)
    
    classification_model = load_classification_model()
    
    if classification_model:
        uploaded_file_class = st.file_uploader("Upload Gambar Makanan (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], key="classify")

        if uploaded_file_class:
            st.divider()
            col_class_input, col_class_output = st.columns(2)

            image_pil = Image.open(uploaded_file_class)
            image_class = image_pil.resize((224, 224))
            
            with col_class_input:
                st.markdown("### Gambar Input (224x224) 📸")
                st.image(image_class, caption="Gambar Diubah Ukuran untuk Model", use_container_width=True)

            if st.button("Klasifikasikan Sekarang 🔍", type="primary", key="classify_btn"):
                with st.spinner("⏳ Mengklasifikasikan gambar dengan ResNet50..."):
                    try:
                        img_array = np.array(image_class)
                        preprocessed_img = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img_array, axis=0))
                        
                        start_time = time.time()
                        predictions = classification_model.predict(preprocessed_img)
                        end_time = time.time()
                        
                        decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=5)[0]
                        
                        # LOGIKA PENENTUAN KLASIFIKASI
                        is_pizza = False
                        # Daftar keyword yang lebih relevan untuk kategori 'Pizza' (termasuk beberapa false positive ResNet50)
                        pizza_keywords = ['pizza', 'cheese_pizza', 'hot_dog', 'bagel', 'focaccia', 'gong_gong'] 
                        main_label = decoded_predictions[0][1].replace('_', ' ').title()
                        
                        # Cek apakah ada keyword "pizza" di Top 5 prediksi
                        for i, (imagenet_id, label, confidence) in enumerate(decoded_predictions):
                            if any(keyword in label.lower() for keyword in pizza_keywords):
                                is_pizza = True
                                
                        # Tampilkan hasil di kolom kanan
                        with col_class_output:
                            st.markdown("### Hasil Klasifikasi AI 🎯")
                            st.markdown(f"**Prediksi Utama:** <span style='color:#ff5722; font-weight: bold;'>{main_label}</span>", unsafe_allow_html=True)
                            
                            st.markdown(f"---")
                            if is_pizza:
                                final_result = "Pizza"
                                st.session_state['classification'] = 'pizza'
                                st.balloons()
                                st.success(f"🎉 Keputusan AI: Objek ini adalah {final_result}! Siap untuk Menu Pizza Spesial!")
                            else:
                                final_result = "Bukan Pizza"
                                st.session_state['classification'] = 'not_pizza'
                                st.info(f"😕 Keputusan AI: Objek ini {final_result}. Kami punya alternatif lezat untuk Anda!")
                                
                            st.markdown(f"<p style='font-size: 1.8rem; text-align: center; font-weight: bold; color: #cc0000;'>KESIMPULAN: {final_result}</p>", unsafe_allow_html=True)
                            st.metric(label="Waktu Pemrosesan", value=f"{end_time - start_time:.3f} s")
                            
                        with st.expander("Lihat Detail Prediksi (Top 5 ImageNet)"):
                            for i, (imagenet_id, label, confidence) in enumerate(decoded_predictions):
                                st.write(f"{i+1}. **{label.replace('_', ' ').title()}** (Keyakinan: {confidence*100:.2f}%)")
                                
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat klasifikasi: {e}")
            else:
                with col_class_output:
                    st.info("Tekan tombol 'Klasifikasikan Sekarang' untuk mendapatkan rekomendasi menu!")
    else:
        st.warning("Model Klasifikasi (ResNet50) tidak dapat dimuat. Pastikan TensorFlow terinstal dengan benar dan koneksi internet stabil.")


# ----------------- MENU REKOMENDASI -----------------
with tabs[3]:
    st.markdown("<h2 class='section-title'>Rekomendasi Menu Personal Pijjahut 🎁</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem;'>Mari kita lihat apa yang paling cocok untuk selera Anda, berdasarkan hasil analisis AI kami di tab sebelumnya! 😋</p>", unsafe_allow_html=True)
    
    menu = {
        'pizza_spesial': [
            {'nama': 'Pizza Margherita Klasik 🇮🇹', 'deskripsi': 'Saus tomat otentik, Mozzarella segar, dan daun Basil. Kesempurnaan Italia. **Pilihan Sempurna**.', 'harga': 'Rp 85.000'},
            {'nama': 'Pizza Pepperoni AI 🔥', 'deskripsi': 'Daging pepperoni premium dengan sentuhan pedas, dijamin terbaik. **Most Popular**.', 'harga': 'Rp 95.000'}
        ],
        'non_pizza_spesial': [
            {'nama': 'Spaghetti Bolognese Pijjahut 🍝', 'deskripsi': 'Spaghetti al dente dengan saus daging rahasia yang kaya rasa. **Alternatif Terbaik**.', 'harga': 'Rp 65.000'},
            {'nama': 'Caesar Salad Segar 🥗', 'deskripsi': 'Salad sehat dengan ayam panggang, crouton, dan dressing Caesar creamy. **Pilihan Sehat**.', 'harga': 'Rp 55.000'}
        ],
        'dessert_minuman': [
            {'nama': 'Lava Cake Cokelat Panas 🍫', 'deskripsi': 'Dessert wajib untuk menutup hidangan Anda. Meleleh di mulut.', 'harga': 'Rp 40.000'},
            {'nama': 'Es Teh Lemon Segar 🍹', 'deskripsi': 'Pendingin yang sempurna setelah makan makanan berat. **Fresh!**', 'harga': 'Rp 20.000'}
        ]
    }
    
    st.divider()
    
    if st.session_state['classification'] == 'pizza':
        st.markdown("""
        <div class='card' style='background: linear-gradient(45deg, #ffe0b2, #ffcc80); border-color: #ff9800; text-align: center;'>
            <p style='font-size: 1.8rem; color: #d84315; font-weight: bold;'>Anda Seorang Pecinta PIZZA Sejati! 👑</p>
            <p style='font-size: 1.2rem;'>Kami tahu selera Anda. Mari tingkatkan pengalaman pizza Anda dengan varian spesial kami atau pendamping sempurna.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.markdown("<h3 style='color:#ff5722;'>🍕 Varian Pizza Wajib Coba</h3>")
            for item in menu['pizza_spesial']:
                st.markdown(f"<div class='menu-item'><span style='font-weight: bold;'>{item['nama']}</span> <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#cc0000; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown("<h3 style='color:#ff5722;'>🍹 Minuman & Dessert Pelengkap</h3>")
            for item in menu['dessert_minuman']:
                st.markdown(f"<div class='menu-item'><span style='font-weight: bold;'>{item['nama']}</span> <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#cc0000; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
                
    elif st.session_state['classification'] == 'not_pizza':
        st.markdown("""
        <div class='card' style='background: linear-gradient(45deg, #e1f5fe, #b3e5fc); border-color: #0288d1; text-align: center;'>
            <p style='font-size: 1.8rem; color: #01579b; font-weight: bold;'>Anda Sedang Mencari Alternatif Lezat! 🥗</p>
            <p style='font-size: 1.2rem;'>Tidak apa-apa! Kami punya menu Italia dan Barat yang tidak kalah enaknya. Coba menu non-pizza andalan kami.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.markdown("<h3 style='color:#ff5722;'>🍝 Pilihan Non-Pizza Terbaik</h3>")
            for item in menu['non_pizza_spesial']:
                st.markdown(f"<div class='menu-item'><span style='font-weight: bold;'>{item['nama']}</span> <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#cc0000; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
                
        with col_rec2:
            st.markdown("<h3 style='color:#ff5722;'>🍰 Dessert untuk Melengkapi</h3>")
            for item in menu['dessert_minuman']:
                st.markdown(f"<div class='menu-item'><span style='font-weight: bold;'>{item['nama']}</span> <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#cc0000; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
        
    else:
        # Pesan untuk meminta klasifikasi
        st.markdown("""
        <div class='card' style='background-color: #fffae6; border-color: #ff9800; border-style: dashed; text-align: center;'>
            <p style='margin: 0; color: #d84315; font-size: 1.3rem;'>
                🚨 <span style='font-weight: bold;'>Lakukan Klasifikasi Gambar (Tab ke-3) terlebih dahulu!</span>
            </p>
            <p style='margin: 0; color: #d84315;'>
                Kami tidak bisa memberikan rekomendasi terbaik tanpa mengetahui selera Anda. Segera unggah foto makanan Anda!
            </p>
        </div>
        """, unsafe_allow_html=True)


# ----------------- KONTAK KAMI -----------------
with tabs[4]:
    st.markdown("<h2 class='section-title'>Hubungi Kami: Layanan Cepat Tanggap 📞</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.2rem; text-align: center;'>Kami selalu siap melayani Anda. Hubungi kami untuk reservasi, pemesanan, atau pertanyaan tentang teknologi AI kami.</p>
        <div class='contact-info'>
            <p><span style='font-weight: bold; color: #ff5722;'>📍 Alamat Eksklusif:</span> Jl. Digitalisasi No. 101, Kota Streamlit, Kode Pos 404 </p>
            <p><span style='font-weight: bold; color: #ff5722;'>📞 Hotline Pemesanan:</span> <a href='tel:02112374992' style='color: #cc0000 !important; text-decoration: none;'>(021) 123-PIZZA (74992)</a></p>
            <p><span style='font-weight: bold; color: #ff5722;'>📧 Dukungan Teknis:</span> <a href='mailto:pijjahut.ai@gmail.com' style='color: #cc0000 !important; text-decoration: none;'>pijjahut.ai@gmail.com</a></p>
            <p><span style='font-weight: bold; color: #ff5722;'>🕒 Jam Operasional:</span> <span style='font-style: italic;'>Buka Setiap Hari, 10:00 - 22:00 WIB</span></p>
        </div>
        
        <br>
        <div style="text-align: center;">
            <p>Ikuti kami di media sosial:</p>
            <p style="font-size: 2rem;">
                <a href="#"><span style="color:#3b5998 !important;">📘</span></a> 
                <a href="#"><span style="color:#e95950 !important;">📸</span></a>
                <a href="#"><span style="color:#00aced !important;">🐦</span></a>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
# ----------------- TENTANG PROYEK -----------------
with tabs[5]:
    st.markdown("<h2 class='section-title'>Tentang Proyek Pijjahut 🎓</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='card' style='background: linear-gradient(135deg, #f0f4c3, #fffde7); border-color: #afb42b;'>
        <p style='font-size: 1.4rem;'>Pijjahut adalah bukti nyata bagaimana <span style='font-weight: bold;'>teknologi kecerdasan buatan</span> dapat diintegrasikan dalam kehidupan sehari-hari, khususnya di industri kuliner.</p>
        
        <hr style="border-color: #afb42b;">
        
        <h3 style='color: #5d4037;'>Detail Teknis Proyek:</h3>
        
        <div class="row">
            <div class="col-md-6 menu-item" style="border-color: #afb42b; margin-top: 10px; text-align: left;">
                <p style='font-weight: bold; color: #cc0000;'>Model Deteksi 🎯</p>
                <ul>
                    <li>**Arsitektur:** YOLO (You Only Look Once)</li>
                    <li>**Tugas:** Deteksi Objek (Piring, Gelas)</li>
                    <li>**File Model:** <span style='font-family: monospace;'>DINI ARIFATUL NASYWA_Laporan 4.pt</span></li>
                </ul>
            </div>
            
            <div class="col-md-6 menu-item" style="border-color: #afb42b; margin-top: 10px; text-align: left;">
                <p style='font-weight: bold; color: #cc0000;'>Model Klasifikasi 🧠</p>
                <ul>
                    <li>**Arsitektur:** ResNet50 (Pre-trained ImageNet)</li>
                    <li>**Tugas:** Klasifikasi Gambar (Pizza vs. Bukan Pizza)</li>
                    <li>**Framework:** TensorFlow/Keras</li>
                </ul>
            </div>
        </div>

        <hr style="border-color: #afb42b;">
        
        <p style='font-size: 1.2rem; text-align: center;'>
            Proyek ini dikembangkan oleh **<span style='font-weight: bold; color: #cc0000;'>Dini Arifatul Nasywa</span>** sebagai inisiatif untuk menampilkan kemampuan <span style='font-weight: bold;'>Deep Learning</span> dalam aplikasi web praktis menggunakan **Streamlit**.
        </p>
        <div style='text-align: center; margin-top: 2rem;'>
            <p style='font-style: italic; color: #cc0000;'>#AIforGood #StreamlitApp #PijjahutProject</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ========================== FOOTER ==========================
st.markdown("---")
st.markdown("<p class='footer'>© 2024 Pijjahut. Dibuat dengan 🍕 dan ❤️ oleh Dini Arifatul Nasywa. Semua hak cipta dilindungi.</p>", unsafe_allow_html=True)
