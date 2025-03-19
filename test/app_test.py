""" Application de test des programmes image_preprocessing, feature_extractor et similarity_search.
!!! : À exécuter depuis le dossier projet via 'python3 test/app_test.py'. """

import logging
from src.image_preprocessing import preprocess_image
from src.feature_extractor import FeatureExtractor
from src.similarity_search import find_top5_categories, find_top5_similar_images, get_image_path

def main(image_path: str):
    """ Fonction principale qui traite une image, extrait ses caractéristiques,
    et trouve les catégories et images similaires.
    :param image_path: Chemin vers l'image à traiter """

    try:
        processed_image = preprocess_image(image_path, target_size=(224, 224), to_tensor=True) # Prétraitement

        extractor = FeatureExtractor() # Extraction du vecteur de caractéristiques
        features = extractor.extract_features(processed_image, from_preprocessed=True)
        print("Vecteur de caractéristiques:")
        print(features)
        print("Dimension du vecteur:", features.shape)

        top5_categories = find_top5_categories(features) # Recherche des 5 catégories les plus similaires
        print("Top 5 catégories les plus similaires :")
        for category, distance in top5_categories:
            print(f"Catégorie: {category}, Distance: {distance}")

        # Recherche des 5 images les plus similaires
        top5_images = find_top5_similar_images(features)
        print("Top 5 images les plus similaires :")
        for index, distance in top5_images:
            print(f"Index: {index}, Distance: {distance}")
            print(f"Chemin de l'image pour l'index {index}: {get_image_path(index)}")

    except Exception as e: # Gestion des erreurs : log l'erreur en cas de problème
        logging.error(f"Erreur lors du traitement de l'image: {e}")

if __name__ == "__main__":
    input_image_path = "ressources/mountain1.png"  # À REMPLIR
    main(input_image_path)