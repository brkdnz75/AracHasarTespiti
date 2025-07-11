import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import sys

# Modellerin yolunu eklemek Streamlit Cloud için gerekmez, çünkü aynı klasörde olacak
# sys.path ekleme kaldırıldı

# Model fonksiyonlarını içe aktar
from fasterrcn import predict as fasterrcnn_predict
from yolov5 import predict as yolov5_predict
from Yolov8 import predict as yolov8_predict

# Sayfa yapılandırması
st.set_page_config(
    page_title="AI Araç Hasar Tespiti",
    layout="wide",
    page_icon="🛠️",
    initial_sidebar_state="collapsed"
)

# CSS ile stil tanımları
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Rubik', sans-serif;
            background-color: #0f0f0f;
            color: #f1f1f1;
        }

        h1 {
            text-align: center;
            color: #ff8800;
            font-size: 3em;
            margin-bottom: 0.5em;
        }

        .stTabs [role="tab"] {
            background-color: #1e1e1e;
            padding: 10px;
            border-radius: 10px;
            margin-right: 5px;
            color: #ddd;
            font-weight: bold;
        }

        .stTabs [aria-selected="true"] {
            background-color: #ff8800;
            color: black;
        }

        .glass-box {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 30px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Başlık ve alt yazı
st.markdown("<h1> Araç Hasar Tespiti | Model Karşılaştırması</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>📷 Tek bir görsel yükleyin, 3 güçlü yapay zeka modeliyle anında kıyaslayın.</p>", unsafe_allow_html=True)

st.markdown("---")

# Görsel yükleme
uploaded_file = st.file_uploader("📤 Lütfen bir görsel yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(temp_path, caption="📥 Yüklenen Görsel", use_container_width=True)
    with col2:
        st.success("✅ Görsel başarıyla yüklendi. Lütfen modellerin çıktıları için aşağıya bakın.")

    st.markdown("---")

    with st.spinner("⚙️ Faster R-CNN çalışıyor..."):
        json1, img1 = fasterrcnn_predict(temp_path)

    with st.spinner("⚙️ YOLOv5 çalışıyor..."):
        json2, img2 = yolov5_predict(temp_path)

    with st.spinner("⚙️ YOLOv8 çalışıyor..."):
        json3, img3 = yolov8_predict(temp_path)

    def convert_cv2_to_pil(cv2_img):
        if isinstance(cv2_img, np.ndarray):
            return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        return None

    st.markdown("## Model Çıktıları")

    tabs = st.tabs(["📘 Faster R-CNN", "📙 YOLOv5", "📗 YOLOv8"])

    for tab, model_name, image, json_data in zip(
        tabs,
        ["Faster R-CNN", "YOLOv5", "YOLOv8"],
        [img1, img2, img3],
        [json1, json2, json3]
    ):
        with tab:
            st.markdown(f"### 🔎 {model_name} Sonucu")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="glass-box">', unsafe_allow_html=True)
                pil_image = convert_cv2_to_pil(image)
                if pil_image:
                    st.image(pil_image, caption=f"{model_name} Anotasyonlu Görsel", use_container_width=True)
                else:
                    st.error("⚠️ Görsel dönüştürülemedi.")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="glass-box">', unsafe_allow_html=True)
                st.markdown("#### 📄 JSON Sonuç")
                st.code(json_data, language="json")
                st.markdown('</div>', unsafe_allow_html=True)

# Tasarımcı kredisi alt footer
st.markdown("""
    <div style='text-align: center; margin-top: 40px;'>
        <a href='https://berkedeniz.com' target='_blank' style='text-decoration: none;'>
            <div style='
                display: inline-block;
                padding: 10px 25px;
                border-radius: 12px;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
                color: #ccc;
                font-size: 0.9em;
                transition: all 0.3s ease;
            ' onmouseover="this.style.background='rgba(255, 136, 0, 0.2)'; this.style.color='#fff';"
               onmouseout="this.style.background='rgba(255, 255, 255, 0.05)'; this.style.color='#ccc';">
                 Design by <strong style='color: #ff8800;'>Berkedeniz</strong>
            </div>
        </a>
    </div>
""", unsafe_allow_html=True)
