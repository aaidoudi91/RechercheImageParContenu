""" Application Streamlit pour la recherche d'images similaires dans le dataset Tiny ImageNet.
Cette application permet :
1. De charger une image depuis l'interface utilisateur.
2. De prétraiter l'image pour l'adapter au modèle.
3. D'extraire les caractéristiques de l'image à l'aide de MobileNetV3.
4. De rechercher les 5 images les plus similaires dans le dataset Tiny ImageNet.
5. D'afficher les images similaires avec leurs distances.

!!! : À éxecuter depuis le repertoire du projet ce fichier avec la commande 'streamlit run src/frontend.py'. """


import streamlit as st, tempfile, os, logging, numpy as np
from PIL import Image
from src.image_preprocessing import preprocess_image
from src.feature_extractor import FeatureExtractor
from src.similarity_search import find_top5_similar_images, get_image_path

st.set_page_config(page_title="L3E1 - CBIR Démo", layout="centered") # Configuration de Streamlit
st.markdown(
    """
    <style>
    header {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
        color: #000000; 
        margin: 0;
        padding: 1;
    }
    .css-1d391kg {
        padding: 1rem;  
    }
    </style>
    """,
    unsafe_allow_html=True,) # Personnalisation du style de la page (CSS)

st.title("L3E1 - CBIR Démo") # Titre de l'application
st.write("Déposez ou sélectionnez une image à analyser.")

TINY_IMAGENET_PATH = "ressources/tiny-imagenet-200"  # Chemin du dataset Tiny ImageNet
EMBEDDINGS_PATH = "ressources/Tiny_ImageNet_Embeddings.npy"  # Embeddings de Tiny ImageNet
CATEGORIES_PATH = "ressources/Tiny_ImageNet_Categories.npy"  # Noms des catégories
if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(CATEGORIES_PATH):
    st.error("Erreur : Les embeddings ou catégories Tiny ImageNet ne sont pas disponibles.")
else:
    embeddings = np.load(EMBEDDINGS_PATH)
    categories = np.load(CATEGORIES_PATH)



uploaded_image = st.file_uploader( # Interface pour charger une image depuis l'utilisateur
    "Choisissez une image (formats acceptés : jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image) # Affichage de l'image sélectionnée
    st.image(image, caption="Image sélectionnée", use_container_width=True)

    st.write("**Nom du fichier :**", uploaded_image.name) # Informations sur le fichier téléchargé
    st.write("**Taille du fichier :**", uploaded_image.size, "octets")

    try: # Sauvegarde temporaire de l'image pour traitement
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_image.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_image.getvalue())
            tmp_file_path = tmp_file.name

        processing_placeholder = st.empty() # Placeholder pour afficher un message pendant le traitement
        processing_placeholder.text("Traitement de l'image en cours...")

        try:
            processed_image = preprocess_image(tmp_file_path, target_size=(224, 224), to_tensor=True) # Prétraitement

            extractor = FeatureExtractor() # Extraction du vecteur de caractéristiques
            features = extractor.extract_features(processed_image, from_preprocessed=True)

            processing_placeholder.empty() # Suppression du message de traitement après extraction des caractéristiques

            top5 = find_top5_similar_images(features) # Recherche des 5 images les plus similaires
            st.subheader("Top 5 des images les plus similaires")  # Affichage des résultats
            col1, col2, col3, col4, col5 = st.columns(5)  # Création de 5 colonnes pour afficher les images
            for i, (index, distance) in enumerate(top5):
                image_path = get_image_path(index)  # Récupération du chemin de l'image similaire
                print(image_path)
                st.write(f"Image {i+1} | Distance : {distance:.4f}") # Affichage des informations
                if os.path.exists(image_path):
                    similar_image = Image.open(image_path).convert("RGB")
                    with [col1, col2, col3, col4, col5][i]: # Affichage dans la colonne correspondante
                        st.image(similar_image, caption=f"Image {i+1}", use_container_width=True)
                else:
                    st.write(f"Image non trouvée : {image_path}")

        except Exception as e:
            processing_placeholder.empty()
            logging.error(f"Erreur lors du traitement de l'image : {e}")
            st.error(f"Erreur lors du traitement de l'image : {e}")

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)