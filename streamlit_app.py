import os
import requests
import streamlit as st
import cv2
from ultralytics import YOLO
import time

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
MODEL_PATH = "yolov8n.pt"

# Télécharger le modèle YOLOv8n si absent
if not os.path.exists(MODEL_PATH):
    with st.spinner("Téléchargement du modèle YOLOv8n..."):
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    st.success("Modèle YOLOv8n téléchargé.")

# Sidebar pour configuration WiFi Tasmota
st.sidebar.image("./images/logo.png", use_container_width=True)
st.sidebar.header("Configurer le WiFi Tasmota")
with st.sidebar.form("wifi_form"):
    tasmota_ip = st.text_input("IP de l'appareil Tasmota", os.getenv("TASMOTA_IP"))
    ssid = st.text_input("Nouveau SSID WiFi")
    password = st.text_input("Mot de passe WiFi", type="password")
    submit = st.form_submit_button("Changer le WiFi")

st.sidebar.info(f"TASMOTA_IP utilisée : {tasmota_ip}")

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

# URL du flux MJPEG (ngrok public)
MJPEG_URL = "https://c19bfdefae9c.ngrok-free.app/cam.mjpeg"

st.title("Détection d'objets")

# Charger le modèle YOLOv8 nano
model = YOLO(MODEL_PATH)

# Capture du flux MJPEG via OpenCV
cap = cv2.VideoCapture(MJPEG_URL)
if not cap.isOpened():
    st.error("Impossible d'ouvrir le flux vidéo. Vérifie l'URL ou la connexion réseau.")
    st.stop()

# Centrage vidéo
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
run = st.checkbox("Démarrer la détection en temps réel")

# Boucle principale compatible Streamlit Cloud
while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Pas de frame reçue, fin du flux ou erreur.")
        break

    # Inference YOLO
    results = model(frame)
    boxes = results[0].boxes
    classes = results[0].names
    keep = []
    for i, box in enumerate(boxes):
        class_id = int(box.cls[0].item())
        class_name = classes[class_id]
        if class_name == "person":  # Le modèle COCO ne détecte pas les visages
            keep.append(i)

    if keep:
        results[0].boxes = boxes[keep]
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame

    # Convertir BGR -> RGB
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Affichage direct dans Streamlit
    frame_placeholder.image(annotated_frame, channels="RGB")

    # Petite pause pour ne pas saturer Streamlit
    time.sleep(0.05)

cap.release()
st.success("Flux vidéo terminé ou arrêté par l'utilisateur.")
