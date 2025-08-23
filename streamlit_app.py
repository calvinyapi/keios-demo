MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
MODEL_PATH = "yolov8n.pt"

# Télécharger le modèle YOLOv8n si absent (pour déploiement Streamlit)
if not os.path.exists(MODEL_PATH):
    with st.spinner("Téléchargement du modèle YOLOv8n..."):
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        st.success("Modèle YOLOv8n téléchargé.")
import os


import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np
import requests
import base64
from io import BytesIO



# Logo dans la sidebar
st.sidebar.image("./images/logo.png", use_container_width=True)
st.sidebar.header("Configurer le WiFi Tasmota")
with st.sidebar.form("wifi_form"):
    tasmota_ip = st.text_input("IP de l'appareil Tasmota", os.getenv("TASMOTA_IP"))
    ssid = st.text_input("Nouveau SSID WiFi")
    password = st.text_input("Mot de passe WiFi", type="password")
    submit = st.form_submit_button("Changer le WiFi")

if submit:
    url = f"http://{tasmota_ip}/cm?cmnd=Backlog+WifiSsid+{ssid};WifiPassword+{password};Restart+1"
    try:
        response = requests.get(url, timeout=5)
        if response.ok:
            st.sidebar.success("Commande envoyée à Tasmota ! L'appareil va redémarrer.")
        else:
            st.sidebar.error("Erreur lors de l'envoi de la commande.")
    except Exception as e:
        st.sidebar.error(f"Erreur : {e}")

# URL du flux MJPEG Tasmota
MJPEG_URL = f"http://{tasmota_ip}:81/stream"

  
st.title("Détection d'objets")

# Charger modèle YOLOv8 nano (rapide)
model = YOLO(MODEL_PATH)


# Capture du flux MJPEG via OpenCV
cap = cv2.VideoCapture(MJPEG_URL)

if not cap.isOpened():
    st.error("Erreur : impossible d'ouvrir le flux vidéo. Vérifie l'URL et la connexion réseau.")
    st.stop()


# Centrage de la vidéo avec du CSS
st.markdown("""
    <style>
    .centered-video {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

frame_placeholder = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("Pas de frame reçue, fin du flux ou erreur.")
        break


    # Inference YOLO
    results = model(frame)

    # Filtrer pour ne garder que les personnes (classe 0) et visages (si le modèle le supporte)
    # Pour YOLOv8 COCO, la classe 0 = person
    # Pour les visages, il faut un modèle spécialisé (voir commentaire ci-dessous)
    boxes = results[0].boxes
    classes = results[0].names
    keep = []
    for i, box in enumerate(boxes):
        class_id = int(box.cls[0].item())
        class_name = classes[class_id]
        if class_name == "person" or class_name == "face":
            keep.append(i)

    # Si on a gardé des détections, on les affiche, sinon on affiche l'image d'origine
    if keep:
        # Annoter uniquement les détections voulues
        results[0].boxes = boxes[keep]
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame

    # Convertir BGR OpenCV -> RGB pour Streamlit
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)


    # Convertir l'image annotée en PNG base64
    buffer = BytesIO()
    import PIL.Image
    PIL.Image.fromarray(annotated_frame).save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    # Afficher l'image centrée et agrandie
    frame_placeholder.markdown(
        f'<div class="centered-video">'
        f'<img src="data:image/png;base64,{img_b64}" style="width:80vw; max-width:100%;"/>'
        f'</div>', unsafe_allow_html=True
    )

    # Petite pause pour laisser le temps à Streamlit d'afficher
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
st.success("Flux vidéo terminé ou arrêté par l'utilisateur.")