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
        body, .stApp {
            background: linear-gradient(135deg, #4e342e 0%, #3e2723 50%, #2c1810 100%); /* gradien coklat tua untuk suasana restaurant */
            color: #fff8e1; /* teks krem terang */
            font-family: 'Arial', sans-serif;
        }

        * {
            color: #fff8e1 !important; /* semua teks jadi krem */
            text-shadow: 1px 1px 2px #3e2723; /* efek timbul */
        }

        .main-title {
            text-align: center;
            font-size: 4rem;
            font-weight: 900;
            color: #fffde7 !important;
            text-shadow: 3px 3px 8px #3e2723;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            animation: fadeIn 2s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .subtitle {
            text-align: center;
            color: #fff8e1 !important;
            font-size: 1.4rem;
            margin-bottom: 2rem;
            font-style: italic;
        }

        .section-title {
            font-size: 2rem;
            font-weight: 700;
            color: #fffde7 !important;
            text-shadow: 2px 2px 4px #3e2723;
            margin-top: 2rem;
            text-align: center;
        }

        /* Button style */
        .stButton > button {
            background: linear-gradient(45deg, #cc0000, #b71c1c) !important;
            color: #fff8e1 !important;
            border-radius: 15px !important;
            font-weight: 700 !important;
            padding: 0.8rem 1.5rem !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background: linear-gradient(45deg, #b71c1c, #8b0000) !important;
            transform: scale(1.05);
            box-shadow: 0 6px 15px rgba(0,0,0,0.7);
        }

        /* Metric cards */
        [data-testid="stMetricValue"], [data-testid="stMetricDelta"], [data-testid="stMetricLabel"] {
            color: #fffde7 !important;
            text-shadow: 1px 1px 3px #3e2723;
        }

        /* Expander text */
        .streamlit-expanderHeader, .streamlit-expanderContent {
            color: #fff8e1 !important;
        }

        /* Upload box */
        [data-testid="stFileUploader"] label {
            color: #fff8e1 !important;
            font-weight: 600;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #fff8e1;
            font-size: 1rem;
            margin-top: 3rem;
            text-shadow: 1px 1px 3px #3e2723;
        }

        /* Divider line */
        hr {
            border: 2px solid #fff8e155;
            border-radius: 5px;
        }

        /* Card style for sections */
        .card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            margin-bottom: 2rem;
        }

        /* Menu recommendation */
        .menu-item {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            text-align: center;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# ========================== HEADER ==========================
st.markdown("<h1 class='main-title'>🍕 Pijjahut 🍕</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Selamat Datang di Restoran Pizza Terbaik! Deteksi Piring & Gelasmu, Klasifikasikan Pizza atau Not Pizza, dan Dapatkan Rekomendasi Menu Spesial! 🔥</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================== NAVIGATION (Sidebar for Restaurant Feel) ==========================
with st.sidebar:
    st.markdown("## 🍴 Navigasi Restoran")
    st.markdown("### Pilih Layanan:")
    nav_option = st.radio("", ["🏠 Beranda", "🔍 Deteksi Objek", "🔮 Klasifikasi Gambar", "🍽️ Menu Rekomendasi", "📞 Kontak Kami"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Tentang Kami")
    st.info("Pijjahut adalah restoran pizza inovatif yang menggunakan AI untuk mendeteksi objek dan mengklasifikasikan makanan. Nikmati pengalaman kuliner yang unik! 🍕❤️")

# ========================== MAIN CONTENT BASED ON NAVIGATION ==========================
if nav_option == "🏠 Beranda":
    st.markdown("<h2 class='section-title'>Selamat Datang di Pijjahut! 🎉</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.2rem;'>Di sini, kami menggabungkan kecanggihan AI dengan cita rasa pizza yang luar biasa! Upload gambar piring atau gelas Anda untuk deteksi objek, atau klasifikasikan apakah itu pizza atau bukan. Berdasarkan hasilnya, kami rekomendasikan menu spesial untuk Anda!</p>
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

elif nav_option == "🔍 Deteksi Objek":
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

elif nav_option == "🔮 Klasifikasi Gambar":
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

                label_map = {0: "Not Pizza 🚫", 1: "Pizza 🍕"}
                result_label = label_map.get(predicted_class, "Unknown")

                st.metric("Kelas Prediksi", result_label)
                st.metric("Tingkat Keyakinan", f"{confidence:.2%}")

                with st.expander("📊 Detail Probabilitas Lengkap"):
                    st.progress(float(predictions[0][0]), text=f"Not Pizza: {predictions[0][0]:.4f}")
                    st.progress(float(predictions[0][1]), text=f"Pizza: {predictions[0][1]:.4f}")
            except Exception as e:
                st.error(f"❌ Oops! Error memuat model ResNet50: {e}")

elif nav_option == "🍽️ Menu Rekomendasi":
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

elif nav_option == "📞 Kontak Kami":
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

# ========================== FOOTER ==========================
st.markdown("---")
st.markdown("<p class='footer'>© 2025 Dini Arifatul Nasywa | Dibuat dengan ❤️ menggunakan Streamlit | Restoran AI Terbaik! 🍕🔥</p>", unsafe_allow_html=True)
