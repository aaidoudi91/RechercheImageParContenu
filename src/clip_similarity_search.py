""" Ce programme utilise le modèle CLIP pour rechercher les images les plus similaires à une requête textuelle parmi
un ensemble d'images pré-calculées de Tiny ImageNet. Il charge les embeddings d'images et permet d'obtenir les
résultats les plus pertinents. """

import torch, clip, os, numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path

# Charger le modèle CLIP
# Vérifie si un GPU est disponible et utilise CUDA si possible, sinon utilise le CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Charger les embeddings et catégories
EMBEDDINGS_PATH = np.load(Path(__file__).parent.parent / "ressources" / "Tiny_ImageNet_CLIP_Embeddings.npy")
CATEGORIES_PATH = np.load(Path(__file__).parent.parent / "ressources" / "Tiny_ImageNet_CLIP_Categories.npy", allow_pickle=True)
# Chemin vers les images Tiny ImageNet
BASE_PATH = Path(__file__).parent.parent / "ressources" / "tiny-imagenet-200" / "train"

def text_to_vector(text):
    """ Encode une requête textuelle en vecteur d'embedding avec CLIP.
        :param text: Texte de la requête.
        :return: Vecteur normalisé représentant la requête. """
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize([text]).to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalisation du vecteur
    return text_features.cpu().numpy()
def find_similar_images(text, top_k=10):
    """ Trouve les top_k images les plus similaires à partir d'une requête textuelle.
        :param text: Texte de la requête.
        :param top_k: Nombre d'images à retourner (par défaut 5).
        :return: Indices des images les plus similaires. """
    query_vector = text_to_vector(text)  # Convertit le texte en vecteur
    distances = cdist(query_vector, EMBEDDINGS_PATH, metric="cosine")[0]  # Calcule les distances cosinus
    return np.argsort(distances)[:top_k]  # Sélectionne les indices des images les plus proches

def get_image_path(index):
    """ Retrouve le chemin de l'image à partir de son index.
    :param index: Index de l'image dans le fichier d'embeddings
    :return: Chemin complet vers l'image """

    class_id = CATEGORIES_PATH[index]  # Récupère la catégorie associée à l'index donné
    class_indices = np.where(CATEGORIES_PATH == class_id)[0]
    image_index = np.where(class_indices == index)[0][0]

    image_name = f"{class_id}_{image_index}.JPEG"
    image_path = os.path.join(BASE_PATH, class_id, "images", image_name)

    return image_path
