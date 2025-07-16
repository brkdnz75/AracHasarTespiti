import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import sys

# ------------------ MODEL İMPORT ------------------
sys.path.append(r"C:\Users\berke\PycharmProjects\fasterrcnn_model")
sys.path.append(r"C:\Users\berke\PycharmProjects\yapayZekaModel")
from fasterrcn import predict as fasterrcnn_predict
from yolov5 import predict as yolov5_predict
from Yolov8 import predict as yolov8_predict

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Vehicle Damage Detection",
    layout="wide",
    page_icon="🛠️",
    initial_sidebar_state="auto"
)

# ------------------ SESSION STATE ------------------
if 'lang' not in st.session_state:
    st.session_state.lang = "tr"

# ------------------ LANGUAGE ------------------
translations = {
    "tr": {
        "title": "Araç Hasar Tespiti | Model Karşılaştırması",
        "subtitle": "📷 Tek bir görsel yükleyin.",
        "upload": "📤 Lütfen bir görsel yükleyin",
        "loaded": "✅ Görsel başarıyla yüklendi. Lütfen modellerin çıktıları için aşağıya bakın.",
        "running": "⚙️ Çalışıyor...",
        "outputs": "## Model Çıktıları",
        "json": "#### 📄 JSON Sonuç",
        "uploaded_image": "📥 Yüklenen Görsel"
    },
    "en": {
        "title": "Vehicle Damage Detection | Model Comparison",
        "subtitle": "📷 Upload a single image.",
        "upload": "📤 Please upload an image",
        "loaded": "✅ Image successfully uploaded. Please see model outputs below.",
        "running": "⚙️ Running...",
        "outputs": "## Model Outputs",
        "json": "#### 📄 JSON Result",
        "uploaded_image": "📥 Uploaded Image"
    }
}

# ------------------ SIDEBAR ------------------

lang_choice = st.sidebar.selectbox("🌐 Language / Dil", ["Türkçe", "English"])
st.session_state.lang = "tr" if lang_choice == "Türkçe" else "en"



text = translations[st.session_state.lang]

# ------------------ CUSTOM CSS ------------------


light_css = """
<style>
body, html, [class*="css"] {
    background-color: #f5f5f5;
    color: #222;
    font-family: 'Rubik', sans-serif;
}
h1, h2, h3, h4 {
    color: #e67e22;
}
.stButton>button {
    background-color: #e67e22;
    color: white;
    border-radius: 8px;
}
.glass-box {
    background: rgba(0, 0, 0, 0.05);
    border-radius: 15px;
    padding: 20px;
    border: 1px solid rgba(0,0,0,0.1);
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 30px rgba(0,0,0,0.1);
}
.stTabs [role="tab"] {
    background-color: #ddd;
    padding: 10px;
    border-radius: 10px;
    margin-right: 5px;
    color: #333;
    font-weight: bold;
}
.stTabs [aria-selected="true"] {
    background-color: #e67e22;
    color: white;
}
</style>
"""



# ------------------ TITLE ------------------
st.markdown(f"<h1>{text['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center;'>{text['subtitle']}</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ FILE UPLOADER ------------------
uploaded_file = st.file_uploader(text['upload'], type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(temp_path, caption=text["uploaded_image"], use_container_width=True)
    with col2:
        st.success(text["loaded"])

    st.markdown("---")

    with st.spinner(text["running"] + " Faster R-CNN"):
        json1, img1 = fasterrcnn_predict(temp_path)

    with st.spinner(text["running"] + " YOLOv5"):
        json2, img2 = yolov5_predict(temp_path)

    with st.spinner(text["running"] + " YOLOv8"):
        json3, img3 = yolov8_predict(temp_path)

    def convert_cv2_to_pil(cv2_img):
        if isinstance(cv2_img, np.ndarray):
            return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        return None

    st.markdown(text["outputs"])
    tabs = st.tabs(["📘 Faster R-CNN", "📙 YOLOv5", "📗 YOLOv8"])

    for tab, model_name, image, json_data in zip(
        tabs,
        ["Faster R-CNN", "YOLOv5", "YOLOv8"],
        [img1, img2, img3],
        [json1, json2, json3]
    ):
        with tab:
            st.markdown(f"### {model_name}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="glass-box">', unsafe_allow_html=True)
                pil_image = convert_cv2_to_pil(image)
                if pil_image:
                    st.image(pil_image, use_container_width=True)
                else:
                    st.error("⚠️ Image conversion failed.")
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="glass-box">', unsafe_allow_html=True)
                st.markdown(text["json"])
                st.code(json_data, language="json")
                st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("""
    <div style='text-align: center; margin-top: 40px;'>
        <a href='https://berkedeniz.com' target='_blank' style='text-decoration: none;'>
            <div style='
                display: inline-block;
                padding: 10px 25px;
                border-radius: 12px;
                background: rgba(255, 136, 0, 0.1);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 136, 0, 0.2);
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
                color: #ccc;
                font-size: 0.9em;
                transition: all 0.3s ease;
            ' onmouseover="this.style.background='rgba(255, 136, 0, 0.2)'; this.style.color='#fff';"
               onmouseout="this.style.background='rgba(255, 136, 0, 0.1)'; this.style.color='#ccc';">
                 Design by <strong style='color: #ff8800;'>Berke Deniz</strong>
            </div>
        </a>
    </div>
""", unsafe_allow_html=True)
