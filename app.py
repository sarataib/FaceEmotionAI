import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="Analyse d'Émotions", layout="wide")

# ---------- CSS ----------
st.markdown("""<style>/* Ton CSS existant ici... */</style>""", unsafe_allow_html=True)

# Dictionnaire émotions
EMOTIONS_FR = {
    0: 'Colère 😠', 
    1: 'Dégoût 🤢', 
    2: 'Peur 😨', 
    3: 'Tristesse 😢', 
    4: 'Surprise 😲', 
    5: 'Joie 😊'
}

@st.cache_resource
def charger_modele():
    return load_model('emotion_recognition_model.h5', compile=False)

def detecter_visage(img):
    img_cv = np.array(img.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return x, y, x+w, y+h
    return None

def ameliorer_qualite(img):
    img_cv = np.array(img.convert('RGB'))
    img_cv = cv2.medianBlur(img_cv, 3)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img_cv = cv2.filter2D(img_cv, -1, kernel)
    return Image.fromarray(img_cv)

def analyser_emotion(img, modele):
    img = img.convert('L').resize((48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=[0, -1]) / 255.0
    predictions = modele.predict(img_array)
    classe_predite = np.argmax(predictions[0])
    confiance = np.max(predictions[0])
    return classe_predite, confiance, predictions[0]

# ---------- INTERFACE ----------
st.title("🔍 Analyse des Émotions Faciales")

option = st.radio("📸 Source de l'image", ["Téléverser une image", "Utiliser la caméra"])

fichier = None
img = None
img_to_analyze = None
zoomed_img = None

if option == "Téléverser une image":
    fichier = st.file_uploader("Choisissez une photo de visage", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if fichier:
        img = Image.open(fichier)
        img_to_analyze = img

elif option == "Utiliser la caméra":
    fichier = st.camera_input("Prenez une photo avec la caméra")
    if fichier:
        img = Image.open(fichier)
        st.subheader("🔍 Ajustement de l'image")
        face_coords = detecter_visage(img)
        auto_zoom = st.checkbox("Zoom automatique sur le visage", value=True)

        if auto_zoom and face_coords:
            zoomed_img = img.crop(face_coords)
            st.success("Visage détecté automatiquement!")
        else:
            if not face_coords:
                st.warning("Aucun visage détecté. Veuillez ajuster manuellement.")
            with st.expander("Contrôles manuels de zoom"):
                col1, col2 = st.columns(2)
                with col1:
                    zoom_level = st.slider("Niveau de zoom", 1.0, 3.0, 1.5, 0.1)
                    x_center = st.slider("Centre X", 0.0, 1.0, 0.5, 0.01)
                    y_center = st.slider("Centre Y", 0.0, 1.0, 0.5, 0.01)
                with col2:
                    width, height = img.size
                    new_width = width / zoom_level
                    new_height = height / zoom_level
                    left = max(0, (width * x_center) - (new_width / 2))
                    top = max(0, (height * y_center) - (new_height / 2))
                    right = min(width, left + new_width)
                    bottom = min(height, top + new_height)
                    zoomed_img = img.crop((left, top, right, bottom))
        
        # Image à analyser améliorée
        if zoomed_img:
            img_to_analyze = ameliorer_qualite(zoomed_img)

# ---------- AFFICHAGE ----------
if img:
    col_img, col_result = st.columns(2)

    with col_img:
        st.image(img, caption="📷 Image originale", width=300)
        if zoomed_img:
            st.image(zoomed_img, caption="🔍 Image zoomée", width=300)

    if st.button("Analyser l'émotion", type="primary"):
        modele = charger_modele()
        idx_emotion, confiance, probs = analyser_emotion(img_to_analyze, modele)
        emotion = EMOTIONS_FR[idx_emotion]

        with col_result:
            st.markdown(f"""
            <div class='result-card'>
                <h2>Résultat : {emotion}</h2>
                <p style='font-size:20px'>Confiance : <b>{confiance:.1%}</b></p>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("📊 Probabilités détaillées :")
            for i, prob in enumerate(probs):
                progress = int(prob * 100)
                st.markdown(f"""
                {EMOTIONS_FR[i]}:
                <progress value='{progress}' max='100'></progress> {prob:.1%}
                """, unsafe_allow_html=True)
