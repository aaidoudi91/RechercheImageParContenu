import streamlit as st
from pathlib import Path

def main():
    st.title("À propos")
        image = Image.open(Path(__file__).parent.parent / "ressources" / "tiny-imagenet-200/train/n07873807/images/n07873807_377.JPEG"
    st.write("Informations sur l'application et l'équipe de développement.")
    # Vérifier si la section "À propos" a déjà été chargée
    if "a_propos_loaded" not in st.session_state:
        st.session_state.a_propos_loaded = True
    # Affichage des informations sur le projet
    st.markdown("""
    Ce projet **L3E1** a été réalisé dans le cadre de la licence - Intelligence Artificielle.
    - 🔍 **Recherche d’images similaires** à l’aide de MobileNetV3
    - ⚡ **Recherche rapide** avec FAISS
    - 📊 **Visualisation avec t-SNE**
    - 👩‍💻 **Réalisé par** Nassilya - Aaron - Mathieu - Julie
    """)
if __name__ == "__main__":
    main()
