import streamlit as st
import cv2
import numpy as np
import tempfile
from sklearn.cluster import KMeans
from phase1.inference import PlayerInference
from phase2.trackhsv import TrackHSV
from phase2.trackkmeans import TrackKMeans
from phase3.hsvclassifier import HSVClassifier
from phase3.kmeansclassifier import KMeansClassifier



# Configuration de la page
st.set_page_config(page_title="Application Streamlit", layout="wide")

# Création des onglets
tab1, tab2 = st.tabs(["📌 Présentation", "⚙️ Démo"])

# Onglet Présentation
with tab1:
    st.title("Bienvenue sur notre Application Streamlit 🎉")
    st.write("""
        Cette application est un exemple simple utilisant **Streamlit**.
        
        Elle contient deux onglets :
        - **Présentation** : Cette section explique brièvement l'application.
        - **Démo** : Une démonstration interactive.
    """)
    st.image("https://source.unsplash.com/random/800x300", caption="Illustration")

# Onglet Démo
with tab2:
    st.title("🎬 Démonstration")

    # Étape 1 : Sélection de la vidéo
    st.header("1️⃣ Sélection de la Vidéo")
    
    col1, col2 = st.columns(2)

    with col1:
        uploaded_video = st.file_uploader("Téléversez une vidéo 📂", type=["mp4", "avi", "mov"])
    
    with col2:
        st.write("Ou sélectionnez une vidéo prédéfinie :")
        video_options = {
            "🏰 Colisée (Rome)": "videos/colosseum.mp4",
            "🏜️ Grand Canyon": "videos/canyon.mp4",
            "🏠 Scène intérieure": "videos/room.mp4"
        }
        selected_video = st.radio("Vidéos disponibles :", list(video_options.keys()))

    # Définition du chemin de la vidéo
    video_path = None
    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_video.read())
            video_path = temp_file.name
    else:
        video_path = video_options[selected_video]

    # Affichage de la vidéo sélectionnée
    if video_path:
        st.video(video_path)

    # Bouton d'inférence
    if st.button("🚀 Lancer l'inférence"):

        st.header("2️⃣ Extraction d'Images Clés")
        st.write("📸 Extraction d'images à partir de la vidéo...")

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        extracted_images = []
        for i in range(5):  # Extraire 5 images à intervalles réguliers
            frame_index = int((frame_count / 5) * i)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                extracted_images.append(frame)

        cap.release()

        if extracted_images:
            st.write("✅ Extraction réussie !")
            col1, col2, col3, col4, col5 = st.columns(5)
            cols = [col1, col2, col3, col4, col5]
            for i, img in enumerate(extracted_images):
                with cols[i]:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Image {i+1}")

        # Étape 3 : Sélection de la méthode de différenciation
        st.header("3️⃣ Méthode de Différenciation")
        method = st.radio("Choisissez une méthode :", ["K-Means", "HSV"], index=0)

        if method == "K-Means":
            st.write("🎨 **K-Means Clustering** pour la segmentation des couleurs.")

            def apply_kmeans(image, k=3):
                """ Applique K-Means Clustering à une image """
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = img.reshape((-1, 3))
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(img)
                clustered_img = kmeans.cluster_centers_[kmeans.labels_]
                clustered_img = clustered_img.reshape(image.shape)
                return clustered_img.astype(np.uint8)

            k_value = st.slider("Nombre de clusters (K) :", 2, 10, 3)
            processed_image = apply_kmeans(extracted_images[0], k=k_value)

        else:
            st.write("🌈 **Segmentation HSV** pour la détection de couleurs.")

            def apply_hsv(image):
                """ Convertit l'image en HSV et applique un masque """
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                lower_bound = np.array([30, 40, 40])
                upper_bound = np.array([80, 255, 255])
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                result = cv2.bitwise_and(image, image, mask=mask)
                return result

            processed_image = apply_hsv(extracted_images[0])

        # Affichage des résultats
        st.write("🔍 **Résultat de la segmentation**")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(extracted_images[0], cv2.COLOR_BGR2RGB), caption="Image Originale")
        with col2:
            st.image(processed_image, caption=f"Image Traitée ({method})")