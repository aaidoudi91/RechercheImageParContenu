""" Module de recherche d'images similaires et de catégorisation avec FAISS
Ce module fournit des fonctions pour :
- Charger les embeddings et les catégories du dataset Tiny ImageNet
- Construire un index FAISS pour la recherche rapide des similarités
- Trouver les 5 catégories les plus similaires à une image donnée
- Trouver les 5 images les plus similaires à une image donnée
- Retrouver le chemin d'une image à partir de son index dans le dataset """

import numpy as np, os, faiss
from scipy.spatial.distance import cosine  # Pour la recherche des catégories uniquement

# Charger les embeddings et catégories Tiny ImageNet
tiny_imagenet_embeddings = np.load("ressources/Tiny_ImageNet_Embeddings.npy").astype('float32')  # faiss nécessite float32
tiny_imagenet_categories = np.load("ressources/Tiny_ImageNet_Categories.npy", allow_pickle=True)

# Charger la correspondance entre identifiants des classes et labels
wordnet_mapping = {}
with open("ressources/tiny-imagenet-200/words.txt", "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            wordnet_mapping[parts[0]] = parts[1]

# ---- Étape 1 : Construire l'index FAISS ---- #
dimension = tiny_imagenet_embeddings.shape[1]  # Taille des vecteurs d'embeddings
index = faiss.IndexFlatL2(dimension)  # Index basé sur la distance L2 (euclidienne)
index.add(tiny_imagenet_embeddings)  # Ajouter les embeddings au moteur FAISS


def find_top5_categories(image_features: np.ndarray):
    """ Trouve les 5 catégories les plus similaires à partir d'un vecteur de caractéristiques.
    :param image_features: Vecteur de caractéristiques de l'image
    :return: Liste des 5 catégories les plus similaires """

    if image_features.shape[0] > tiny_imagenet_embeddings.shape[1]:
        image_features = image_features[:tiny_imagenet_embeddings.shape[1]]
    elif image_features.shape[0] < tiny_imagenet_embeddings.shape[1]:
        image_features = np.pad(image_features, (0, tiny_imagenet_embeddings.shape[1] - image_features.shape[0]))

    distances = [cosine(image_features, emb) for emb in tiny_imagenet_embeddings]  # Calculer la distance cosine
    top_5_indices = np.argsort(distances)[:5]  # Récupérer les indices des 5 catégories les plus proches

    top_5_categories = [(wordnet_mapping.get(tiny_imagenet_categories[idx], "Inconnu"), distances[idx])
                        for idx in top_5_indices]

    return top_5_categories


def find_top5_similar_images(image_features: np.ndarray):
    """ Trouve les 5 images les plus similaires à partir d'un vecteur de caractéristiques avec FAISS.
    :param image_features: Vecteur de caractéristiques de l'image
    :return: Liste des indices des 5 images les plus similaires et leurs distances """

    image_features = image_features.astype('float32').reshape(1, -1)  # Convertir en float32 et ajouter une dimension batch (faster)
    distances, indices = index.search(image_features, 5)  # Rechercher les 5 images les plus proches

    top_5_similar = [(idx, distances[0][i]) for i, idx in enumerate(indices[0])]
    return top_5_similar


def get_image_path(index, base_path="ressources/tiny-imagenet-200"):
    """ Retrouve le chemin de l'image à partir de son index.
    :param index: Index de l'image dans le fichier d'embeddings
    :param base_path: Chemin de base vers le dataset Tiny ImageNet
    :return: Chemin complet vers l'image """

    class_id = tiny_imagenet_categories[index]
    class_indices = np.where(tiny_imagenet_categories == class_id)[0]
    image_index = np.where(class_indices == index)[0][0]

    image_name = f"{class_id}_{image_index}.JPEG"
    image_path = os.path.join(base_path, "train", class_id, "images", image_name)

    return image_path

