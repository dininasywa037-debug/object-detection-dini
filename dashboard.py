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
        @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap'); /* Font menarik untuk judul */

        body, .stApp {
            background: linear-gradient(135deg, #fff8dc 0%, #f5f5dc 50%, #ede0c8 100%); /* gradien krem seperti tepung pizza */
            color: #3e2723; /* teks coklat tua untuk kontras */
            font-family: 'Arial', sans-serif;
        }

        * {
            color: #3e2723 !important; /* semua teks jadi coklat tua */
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5); /* efek timbul dengan bayangan putih */
        }

        .main-title {
            text-align: center;
            font-size: 5rem; /* lebih besar */
            font-weight: 900;
            font-family: 'Pacifico', cursive; /* font menarik */
            color: #cc0000 !important; /* merah untuk judul agar menonjol */
            text-shadow: 4px 4px 10px rgba(0,0,0,0.3), 0 0 20px rgba(204,0,0,0.5); /* efek timbul dan glow */
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            animation: bounceIn 2s ease-in-out;
        }

        @keyframes bounceIn {
            0% { transform: scale(0.3); opacity: 0; }
            50% { transform: scale(1.05); }
            70% { transform: scale(0.9); }
            100% { transform: scale(1); opacity: 1; }
        }

        .subtitle {
            text-align: center;
            color: #5d4037 !important; /* coklat lebih gelap */
            font-size: 1.6rem;
            margin-bottom: 2rem;
            font-style: italic;
            text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.7);
        }

        .section-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #cc0000 !important; /* merah untuk judul section */
            text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
            margin-top: 2rem;
            text-align: center;
            font-family: 'Pacifico', cursive;
        }

        /* Button style */
        .stButton > button {
            background: linear-gradient(45deg, #ff5722, #e64a19) !important; /* oranye-merah untuk tombol */
            color: #fff !important;
            border-radius: 20px !important;
            font-weight: 700 !important;
            padding: 1rem 2rem !important;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            transition: all 0.4s ease;
            font-size: 1.1rem;
        }

        .stButton > button:hover {
            background: linear-gradient(45deg, #e64a19, #bf360c) !important;
            transform: scale(1.1);
            box-shadow: 0 8px 20px rgba(0,0,0,0.5);
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
            font-size: 1.1rem;
            margin-top: 3rem;
            text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.7);
        }

        /* Divider line */
        hr {
            border: 3px solid #d7ccc8;
            border-radius: 10px;
        }

        /* Card style for sections */
        .card {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
            margin-bottom: 2rem;
            border: 2px solid #d7ccc8;
        }

        /* Menu recommendation */
        .menu-item {
            background: rgba(255, 248, 220, 0.9);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            text-align: center;
            font-weight: 600;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }

        .menu-item:hover {
            transform: translateY(-5px);
        }

        /* Tabs style */
        .stTabs [data-baseweb="tab-list"] {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            padding: 0.5rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }

        .stTabs [data-baseweb="tab"] {
            color: #3e2723 !important;
            font-weight: 600;
            font-size: 1.1rem;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #ff5722 !important;
            color: #fff !important;
            border-radius: 10px;
        }

        /* Image styling */
        img {
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
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
        <p style='font-size: 1.3rem;'>Di sini, kami menggabungkan kecanggihan AI dengan cita rasa pizza yang luar biasa! Upload gambar piring atau gelas Anda untuk deteksi objek, atau klasifikasikan apakah itu pizza atau bukan. Berdasarkan hasilnya, kami rekomendasikan menu spesial untuk Anda!</p>
        <p>🍕 <strong>Fitur Utama:</strong></p>
        <ul>
            <li>🔍 Deteksi Piring & Gelas dengan YOLO</li>
            <li>🔮 Klasifikasi Pizza/Not Pizza dengan ResNet50</li>
            <li>🍽️ Rekomendasi Menu Personal</li>
        </ul>
        <p>Jangan ragu untuk eksplorasi! 🚀</p>
    </div>
    """, unsafe_allow_html=True)
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
                st.image(result_img, caption="🎯 Hasil Deteksi Objek", use_container_width=True)
                st.subheader("📋 Detail Deteksi")
                boxes = results[0].boxes
                if len(boxes) > 0:
                    cols = st.columns(3)
                    for i, box in enumerate(boxes):
                        with cols[i % 3]:
                            st.metric(
                                f"Objek {i+1} 🍴",
                                yolo_model.names[int(box.cls)],
                                f"{box.conf[0]:.1%}"
                            )
                else:
                    st.info("ℹ️ Tidak ada objek terdeteksi. Coba gambar yang lebih jelas! 😊")
            except Exception as e:
                st.error(f"❌ Oops! Error memuat model YOLO: {e}")

with tabs[2]:  # 🔮 Klasifikasi Gambar
    st.markdown("<h2 class='section-title'>Klasifikasi: Pizza atau Bukan? 🔮</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p>Apakah gambar ini pizza atau bukan? AI kami akan memberitahu Anda dengan akurat! Upload gambar apa saja dan lihat hasilnya. Jika pizza, kami punya rekomendasi spesial! 🍕</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file_tf = st.file_uploader("📤 Upload Gambar untuk Klasifikasi", type=["jpg", "jpeg", "png"], key="tf")

    if uploaded_file_tf:
        image_tf = Image.open(uploaded_file_tf)
        st.image(image_tf, caption="📷 Gambar Input Anda", use_container_width=True)

        if st.button("🔮 Prediksi Sekarang!", type="primary", key="predict_tf"):
            try:
                tf_model = tf.keras.models.load_model('model/BISMILLAHDINI2_Laporan2.h5', compile=False)
                img_array = np.array(image_tf.resize((128, 128)))
                if img_array.shape[-1] == 4:
                    img_array = img_array[:, :, :3]
                img_array = np.expand_dims(img_array, axis=0)
                img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

                predictions = tf_model.predict(img_array, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class]

                # Simpan hasil klasifikasi ke session state untuk rekomendasi
                st.session_state.predicted_class = predicted_class

                label_map = {0: "Not Pizza 🚫", 1: "Pizza 🍕"}
                result_label = label_map.get(predicted_class, "Unknown")

                st.metric("Kelas Prediksi", result_label)
                st.metric("Tingkat Keyakinan", f"{confidence:.2%}")

                with st.expander("📊 Detail Probabilitas Lengkap"):
                    st.progress(float(predictions[0][0]), text=f"Not Pizza: {predictions[0][0]:.4f}")
                    st.progress(float(predictions[0][1]), text=f"Pizza: {predictions[0][1]:.4f}")
            except Exception as e:
                st.error(f"❌ Oops! Error memuat model ResNet50: {e}")

with tabs[3]:  # 🍽️ Menu Rekomendasi
    st.markdown("<h2 class='section-title'>Rekomendasi Menu Spesial 🍽️</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p>Berdasarkan hasil klasifikasi Anda, kami rekomendasikan menu pizza terbaik! Jika hasilnya 'Pizza', nikmati varian spesial kami. Jika 'Not Pizza', coba hidangan alternatif yang lezat!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simulasi rekomendasi berdasarkan hasil klasifikasi (gunakan session state untuk menyimpan hasil)
    if 'predicted_class' not in st.session_state:
        st.session_state.predicted_class = None
    
    if st.session_state.predicted_class is not None:
        if st.session_state.predicted_class == 1:  # Pizza
            st.markdown("<div class='menu-item'>🍕 <strong>Pizza Margherita</strong> - Keju mozzarella segar, saus tomat, basil. Harga: Rp 50.000</div>", unsafe_allow_html=True)
            st.markdown("<div class='menu-item'>🍕 <strong>Pizza Pepperoni</strong> - Pepperoni gurih, keju cheddar. Harga: Rp 60.000</div>", unsafe_allow_html=True)
        else:  # Not Pizza
            st.markdown("<div class='menu-item'>🍝 <strong>Pasta Carbonara</strong> - Pasta dengan bacon dan telur. Harga: Rp 45.000</div>", unsafe_allow_html=True)
            st.markdown("<div class='menu-item'>🥗 <strong>Salad Caesar</strong> - Salad segar dengan dressing khas. Harga: Rp 35.000</div>", unsafe_allow_html=True)
        st.markdown("### Pesan Sekarang! 📞 Hubungi kami di +62 123 456 789")
    else:
        st.info("🔮 Lakukan klasifikasi gambar terlebih dahulu untuk mendapatkan rekomendasi menu!")

with tabs[4]:  # 📞 Kontak Kami
    st.markdown("<h2 class='section-title'>Hubungi Kami 📞</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p><strong>Alamat:</strong> Jl. Pizza Lovers No. 123, Kota Kuliner</p>
        <p><strong>Telepon:</strong> +62 123 456 789</p>
        <p><strong>Email:</strong> info@pijjahut.com</p>
        <p><strong>Jam Operasional:</strong> 10:00 - 22:00 WIB</p>
        <p>Ikuti kami di sosial media: 🍕 @PijjahutID</p>
    </div>
    """, unsafe_allow_html=True)

with tabs[5]:  # ℹ️ Tentang Kami
    st.markdown("<h2 class='section-title'>Tentang Kami ℹ️</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.3rem;'>Pijjahut adalah restoran pizza inovatif yang menggunakan AI untuk mendeteksi objek dan mengklasifikasikan makanan. Kami percaya bahwa teknologi dapat meningkatkan pengalaman kuliner Anda!</p>
        <p>🍕 <strong>Misi Kami:</strong> Menyediakan pizza berkualitas tinggi dengan sentuhan kecanggihan AI.</p>
        <p>❤️ <strong>Tim Kami:</strong> Dibuat oleh Dini Arifatul Nasywa dengan cinta dan dedikasi.</p>
        <p>🚀 Bergabunglah dengan kami untuk pengalaman kuliner yang unik dan tak terlupakan!</p>
    </div>
    """, unsafe_allow_html=True)

# ========================== FOOTER ==========================
st.markdown("---")
st.markdown("<p class='footer'>© 2025 Dini Arifatul Nasywa | Dibuat dengan ❤️ menggunakan Streamlit | Restoran AI Terbaik! 🍕🔥</p>", unsafe_allow_html=True)
