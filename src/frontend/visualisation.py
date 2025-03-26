import streamlit as st
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    st.title("Visualisation")
    st.write("Projection t-SNE des embeddings")
    # Vérifier si la visualisation t-SNE est déjà stockée en session
    if "tsne_data" not in st.session_state:
        # Générer et stocker la visualisation dans la session
        x = np.random.randn(100)
        y = np.random.randn(100)
        fig, ax = plt.subplots()
        ax.scatter(x, y, alpha=0.6)
        st.session_state.tsne_data = fig  # Stocker l’image dans la session
    # Afficher l’image de t-SNE stockée
    st.pyplot(st.session_state.tsne_data)
    # Charger et afficher l'image générée
    image_path = os.path.join("visualization", "tsne_plot.png")

    if os.path.exists(image_path):
        st.image(Image.open(image_path), caption="Projection t-SNE", use_container_width=True)
    else:
        st.error("L'image t-SNE n'a pas été trouvée.")


if __name__ == "__main__":
    main()
