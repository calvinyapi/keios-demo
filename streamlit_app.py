import os
import requests
import streamlit as st
import cv2
from ultralytics import YOLO
import time

# --------------------------
# Config réseau / appareils
# --------------------------
CAM_IP = "192.168.1.89"      # 👈 IP de la caméra (flux vidéo)
TASMOTA_IP = "192.168.1.50"  # 👈 IP pour les commandes Tasmota (si différente)

# --------------------------
# Modèle YOLO
# --------------------------
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
MODEL_PATH = "yolov8n.pt"

# --------------------------
# UI
# --------------------------
st.sidebar.image("./images/logo.png", width="stretch")  # (fix streamlit)
st.sidebar.header("Configurer le WiFi Tasmota")

with st.sidebar.form("wifi_form"):
    ssid = st.text_input("Nouveau SSID WiFi")
    password = st.text_input("Mot de passe WiFi", type="password")
    submit = st.form_submit_button("Changer le WiFi")

st.sidebar.info(f"TASMOTA_IP utilisée : {TASMOTA_IP}")

if submit:
    url = f"http://{TASMOTA_IP}/cm?cmnd=Backlog+WifiSsid+{ssid};WifiPassword+{password};Restart+1"
    try:
        r = requests.get(url, timeout=5)
        st.sidebar.success("Commande envoyée, l'appareil redémarre." if r.ok else "Erreur lors de l'envoi.")
    except Exception as e:
        st.sidebar.error(f"Erreur : {e}")

st.title("Détection d'objets (flux local MJPEG)")

# --------------------------
# Téléchargement du modèle
# --------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Téléchargement du modèle YOLOv8n..."):
        r = requests.get(MODEL_URL, timeout=30)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    st.success("Modèle YOLOv8n téléchargé.")

# Charger YOLO
model = YOLO(MODEL_PATH)

# --------------------------
# Ouverture robuste du flux
# --------------------------
# Essaye plusieurs endpoints de la cam (MJPEG & RTSP)
CANDIDATES = [
    f"http://{CAM_IP}:81/stream",      # MJPEG (Option 1 Tasmota WcStream 1; WcPort 81)
    f"http://{CAM_IP}/stream",         # MJPEG sur port 80
    f"http://{CAM_IP}/cam.mjpeg",      # Variante MJPEG
    f"rtsp://{CAM_IP}:8554/mjpeg/1",   # RTSP si activé
]

# Timeouts OpenCV (µs) pour éviter les blocages
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|rw_timeout;5000000|max_delay;5000000"

def try_open(url: str):
    """Tente d'ouvrir une URL avec FFMPEG puis sans; retourne cap ou None."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        return cap
    cap.release()
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        return cap
    cap.release()
    return None

def open_stream(candidates, tries=15, pause=1.5):
    """Essaie plusieurs URLs pendant un certain temps."""
    info = st.empty()
    for t in range(tries):
        for url in candidates:
            info.info(f"Connexion au flux… (essai {t+1}/{tries})\n{url}")
            cap = try_open(url)
            if cap is not None:
                info.success(f"Flux ouvert : {url}")
                return cap, url
        time.sleep(pause)  # la cam peut mettre 15–30 s à être prête après reboot
    info.empty()
    return None, None

cap, OPEN_URL = open_stream(CANDIDATES)
if cap is None:
    st.error("Impossible d'ouvrir le flux (port fermé / URL erronée / service désactivé). "
             "Vérifie WcStream/WcPort ou active RTSP.")
    st.stop()

# --------------------------
# Contrôles UI
# --------------------------
st.caption(f"Source: {OPEN_URL}")
run = st.checkbox("Démarrer la détection en temps réel", value=False)
conf = st.slider("Seuil de confiance YOLO", 0.1, 0.9, 0.35, 0.05)

frame_placeholder = st.empty()

# --------------------------
# Boucle principale (avec auto-reconnect)
# --------------------------
def reconnect():
    """Ferme et rouvre le flux si coupure."""
    try:
        cap.release()
    except Exception:
        pass
    new_cap, new_url = open_stream(CANDIDATES, tries=10, pause=1.5)
    return new_cap, new_url

while run:
    ok, frame = cap.read()
    if not ok or frame is None:
        st.warning("Flux coupé… tentative de reconnexion.")
        cap, OPEN_URL = reconnect()
        if cap is None:
            st.error("Impossible de se reconnecter au flux.")
            break
        else:
            st.info(f"Reconnecté à {OPEN_URL}")
            continue

    # Inference YOLO — on ne garde que la classe 'person' (COCO id=0)
    results = model(frame, classes=[0], conf=conf, verbose=False)
    annotated_frame = results[0].plot()

    # BGR -> RGB pour Streamlit
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    frame_placeholder.image(annotated_frame, channels="RGB")

    # Limiter le débit d'affichage (respire un peu)
    time.sleep(0.03)

try:
    cap.release()
except Exception:
    pass

st.success("Flux vidéo terminé ou arrêté par l'utilisateur.")
