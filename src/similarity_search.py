""" Module de recherche d'images similaires et de catégorisation avec FAISS
Ce module fournit des fonctions pour :
- Charger les embeddings et les catégories du dataset Tiny ImageNet
- Construire un index FAISS pour la recherche rapide des similarités
- Trouver les 5 catégories les plus similaires à une image donnée
- Trouver les 5 images les plus similaires à une image donnée
- Retrouver le chemin d'une image à partir de son index dans le dataset """

import numpy as np, os, faiss
from scipy.spatial.distance import cosine  # Pour la recherche des catégories uniquement
from pathlib import Path

# Charger les embeddings et catégories Tiny ImageNet
TINY_IMAGENET_PATH = Path(__file__).parent.parent / "ressources" / "tiny-imagenet-200"
EMBEDDINGS_PATH =  np.load(Path(__file__).parent.parent / "ressources"/"Tiny_ImageNet_MobilNetV3_Embeddings.npy").astype('float32')
CATEGORIES_PATH = np.load(Path(__file__).parent.parent / "ressources"/"Tiny_ImageNet_MobilNetV3_Categories.npy", allow_pickle=True)

# Construire l'index FAISS
dimension = EMBEDDINGS_PATH.shape[1]  # Taille des vecteurs d'embeddings
index = faiss.IndexFlatL2(dimension)  # Index basé sur la distance L2 (euclidienne)
index.add(EMBEDDINGS_PATH)  # Ajouter les embeddings au moteur FAISS


def find_top5_categories(image_features: np.ndarray):
    """ Trouve les 5 catégories les plus similaires à partir d'un vecteur de caractéristiques.
    :param image_features: Vecteur de caractéristiques de l'image
    :return: Liste des 5 catégories les plus similaires """

    # Charger la correspondance entre identifiants des classes et labels
    wordnet_mapping = {}
    with open(TINY_IMAGENET_PATH / "words.txt", "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                wordnet_mapping[parts[0]] = parts[1]

    if image_features.shape[0] > EMBEDDINGS_PATH.shape[1]:
        image_features = image_features[:EMBEDDINGS_PATH.shape[1]]
    elif image_features.shape[0] < EMBEDDINGS_PATH.shape[1]:
        image_features = np.pad(image_features, (0, EMBEDDINGS_PATH.shape[1] - image_features.shape[0]))

    distances = [cosine(image_features, emb) for emb in EMBEDDINGS_PATH]  # Calculer la distance cosine
    top_5_indices = np.argsort(distances)[:5]  # Récupérer les indices des 5 catégories les plus proches

    top_5_categories = [(wordnet_mapping.get(CATEGORIES_PATH[idx], "Inconnu"), distances[idx])
                        for idx in top_5_indices]

    return top_5_categories


def find_top_similar_images(image_features: np.ndarray, k):
    """ Trouve les k images les plus similaires à partir d'un vecteur de caractéristiques avec FAISS.
    :param image_features: Vecteur de caractéristiques de l'image
    :return: Liste des indices des k images les plus similaires et leurs distances """

    #Vérifier la dimension de l'image avant la recherche FAISS
    print(f"Dimension réelle de l'image en entrée: {image_features.shape}")
    print(f"Dimension attendue par FAISS: {index.d}")

    # Assurer que l'image est bien un vecteur 1D
    image_features = image_features.astype('float32').reshape(1, -1)

    #Vérifier la taille avant la recherche
    if image_features.shape[1] != index.d:
        print(f"Erreur : la dimension de l'image ({image_features.shape[1]}) ne correspond pas à la dimension FAISS ({index.d})")

    #Lancer la recherche FAISS
    distances, indices = index.search(image_features, k)
    top_k_similar = [(idx, distances[0][i]) for i, idx in enumerate(indices[0])]

    return top_k_similar


def get_image_path(index_image, base_path=TINY_IMAGENET_PATH):
    """ Retrouve le chemin de l'image à partir de son index.
    :param index_image: Index de l'image dans le fichier d'embeddings
    :param base_path: Chemin de base vers le dataset Tiny ImageNet
    :return: Chemin complet vers l'image """

    class_id = CATEGORIES_PATH[index_image]
    class_indices = np.where(CATEGORIES_PATH == class_id)[0]
    image_index = np.where(class_indices == index_image)[0][0]

    image_name = f"{class_id}_{image_index}.JPEG"
    image_path = os.path.join(base_path, "train", class_id, "images", image_name)

    return image_path
