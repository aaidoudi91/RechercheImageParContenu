import streamlit as st
import tempfile
import os, sys
import traceback
from PIL import Image
from image_preprocessing import preprocess_image
from feature_extractor import FeatureExtractor
from similarity_search import find_top5_similar_images, get_image_path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def main():
    # Initialisation sécurisée
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    # Configuration UI
    st.markdown(
        """
        <style>
        .title, .subtitle, .description {
        color: black !important;
        font-weight: bold;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #FAE5D3;
            font-family: 'Arial', sans-serif;
        }
        .header {
            background-color: #E8C3A6;
            padding: 20px;
            text-align: left;
            font-size: 24px;
            font-weight: bold;
            color: black;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .main-box {
            background-color: black;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            width: 80%;
            margin: auto;
        }
        .upload-box {
            background-color: #F5E6DA;
            padding: 40px;
            border-radius: 12px;
            text-align: center;
            border: 2px dashed #D2691E;
            margin-bottom: 10px;
        }
        .upload-box img {
            width: 60px;
            opacity: 0.7;
        }
        .upload-box p {
            color: black;
            font-size: 16px;
            font-weight: bold;
        }
        .upload-btn {
            background-color: #E8C3A6;
            color: black;
            padding: 10px 20px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            display: inline-block;
            margin-top: 10px;
        }
        .upload-btn:hover {
            background-color: #D9B08C;
        }
        .steps-container {
            text-align: center;
            padding: 20px;
        }
        .step {
            font-size: 18px;
            font-weight: bold;
            margin: 10px;
            color: #D2691E;
        }
        .step-number {
            font-size: 24px;
            font-weight: bold;
            color: #FF8C00;
        }
        div[data-testid="stFileUploader"] label {
            color: #FF8C00 !important;
            font-weight: bold;
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # TITRES
    st.markdown('<h1 class="title">Groupe de Projet L3E1</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subtitle">L’IA qui interprète le contenu visuel</h2>', unsafe_allow_html=True)
    st.markdown("""
        <p style="color: black; font-size: 16px;">
            Le site dédié à la recherche d'images par le contenu, optimisé par l’IA. 
            Grâce aux avancées du Deep Learning et de la vision par ordinateur, 
            les caractéristiques visuelles de votre image sont analysées en profondeur 
            pour offrir une recherche plus intelligente et précise.
        </p>
    """, unsafe_allow_html=True)
    # Visualisation t-SNE
    st.subheader("Visualisation t-SNE des embeddings")
    st.write("Exploration visuelle des vecteurs de caractéristiques du dataset CIFAR-10.")
    #if st.button("Générer la visualisation t-SNE"):
        #fig = generate_tsne_plot()
        #st.pyplot(fig)
    # Étapes
    st.markdown('<div class="steps-container">', unsafe_allow_html=True)
    st.markdown('<div class="step"><span class="step-number">Étape 1</span><br>Déposez votre image</div>', unsafe_allow_html=True)
    st.markdown('<div class="step"><span class="step-number">Étape 2</span><br>L’IA cherche des similarités</div>', unsafe_allow_html=True)
    st.markdown('<div class="step"><span class="step-number">Étape 3</span><br>Découvrez les images assimilées</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    # Upload
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h2 class="subtitle">Effectuer une recherche</h2>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Faites glisser une image ici ou importez un fichier", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.session_state.uploaded_image = uploaded_image
    if st.session_state.uploaded_image is not None:
        image = Image.open(st.session_state.uploaded_image)
        st.image(image, caption="Image sélectionnée", use_container_width=True)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_image.name.split('.')[-1]) as tmp_file:
                tmp_file.write(uploaded_image.getvalue())
                tmp_file_path = tmp_file.name
            st.markdown('<div class="upload-box">Traitement de l\'image en cours...</div>', unsafe_allow_html=True)
            processed_image = preprocess_image(tmp_file_path, target_size=(224, 224), to_tensor=True)
            extractor = FeatureExtractor()
            features = extractor.extract_features(processed_image, from_preprocessed=True)
            top5 = find_top5_similar_images(features)
            st.markdown('<h2 class="subtitle">Images assimilées</h2>', unsafe_allow_html=True)
            col1, col2, col3, col4, col5 = st.columns(5)
            for i, (index, distance) in enumerate(top5):
                image_path = get_image_path(index)
                if os.path.exists(image_path):
                    similar_image = Image.open(image_path).convert("RGB")
                    with [col1, col2, col3, col4, col5][i]:
                        st.image(similar_image, caption=f"Image {i+1}", use_container_width=True)
                else:
                    st.write(f"Image non trouvée : {image_path}")
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : {e}")
            st.text(traceback.format_exc())
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    st.markdown('</div>', unsafe_allow_html=True)
# Lancer la page
if __name__ == "__main__":
    main()
