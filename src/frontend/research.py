import streamlit as st
import tempfile
import os, sys
import traceback
from PIL import Image
from src.image_preprocessing import preprocess_image
from src.feature_extractor import FeatureExtractor
from src.similarity_search import find_top_similar_images, get_image_path
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
        #MainMenu {
        visibility: hidden;
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
            padding: 10px 10px;
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
            padding: 10px;
        }
        .step {
            font-size: 18px;
            font-weight: bold;
            margin: 10px;
            color: #D2691E;
            display: inline-block;
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
        .similar-image-container {
        margin-bottom: 15px; /* Espace vertical entre les images */
        padding: 10px; /* Espace intérieur autour de l'image */
        }

        </style>
        """,unsafe_allow_html=True)

    # Titre de la page
    st.markdown('<h2 class="subtitle">The AI that interprets visual content</h2>', unsafe_allow_html=True)

    #Description
    st.markdown('<h4 class="subtitle">🏞️ Content-Based Image Retrieval (CBIR)</h4>', unsafe_allow_html=True)
    st.markdown("""<p style="color: black; font-size: 16px;">
            Content-Based Image Retrieval (CBIR) is a technique that allows users to search for visually similar images 
            based on their content rather than metadata or keywords. Unlike traditional search methods that rely on text 
            annotations, CBIR analyzes the actual visual features of an image—such as texture, shape, and color—to find 
            matches in a large database.</p>""", unsafe_allow_html=True)
    st.markdown('<h4 class="subtitle">💡 How Does It Work?</h4>', unsafe_allow_html=True)
    st.markdown("""<p style="color: black; font-size: 16px;">
        • Upload an image that you want to find similar results for.<br>
        • The AI extracts deep visual features such as shapes, textures, and colors.<br>
         • The system searches through the image database to retrieve the most similar images.<br> 
         """, unsafe_allow_html=True)
    st.markdown('<h4 class="subtitle">🔍 Why Use CBIR?</h4>', unsafe_allow_html=True)
    st.markdown("""<p style="color: black; font-size: 16px;">
        • No need for manual tagging or metadata—images are retrieved purely based on their visual content.<br>
        • Faster and more accurate searches, thanks to deep learning models optimized for visual recognition.<br>
        • Supports both image-based and text-based queries, allowing flexible ways to find relevant content.</p>
        """, unsafe_allow_html=True)

    # Étapes
    st.markdown("""
        <div class="steps-container">
            <span class="step"><span class="step-number">Step 1</span><br>Upload your image</span> 
            <span class="step"><span class="step-number">➝</span><br></span>
            <span class="step"><span class="step-number">Step 2</span><br>AI looks for similarities </span>
            <span class="step"><span class="step-number">➝</span><br></span>
            <span class="step"><span class="step-number">Step 3</span><br>Discover the assimilated images</span>
        </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="section">', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.session_state.uploaded_image = uploaded_image
    if st.session_state.uploaded_image is not None:
        image = Image.open(st.session_state.uploaded_image)
        st.image(image, width=300)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_image.name.split('.')[-1]) as tmp_file:
                tmp_file.write(uploaded_image.getvalue())
                tmp_file_path = tmp_file.name

            processed_image = preprocess_image(tmp_file_path, target_size=(224, 224), to_tensor=True)
            extractor = FeatureExtractor()
            features = extractor.extract_features(processed_image, from_preprocessed=True)
            top = find_top_similar_images(features, 12)

            st.markdown('<h2 class="subtitle">Assimilated images</h2>', unsafe_allow_html=True)
            cols = st.columns(3)

            for i, (index, distance) in enumerate(top):
                image_path = get_image_path(index)
                if os.path.exists(image_path):
                    similar_image = Image.open(image_path).convert("RGB")
                    with cols[i % 3]:
                        st.markdown('<div class="similar-image-container">', unsafe_allow_html=True)
                        st.image(similar_image, use_container_width=True)
                        st.markdown(
                            f'<p style="text-align: center; color: black;;">Image {i} - Distance: {distance:.4f}</p>',
                            unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
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