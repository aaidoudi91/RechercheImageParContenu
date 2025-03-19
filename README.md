# Projet L3E1 : Recherche d'image par similarité et indexation d'une base d'image
Ce projet implémente un système de recherche d'images similaires basé sur l'extraction de caractéristiques et l'indexation efficace avec FAISS. Il est conçu pour des applications telles que la recherche d'images, la recommandation de contenu et l'analyse basée sur le contenu visuel.

## Modules du projet
- **Prétraitement des images** : Via *image_preprocessing.py*, le module prépare les images pour l'entrée dans un réseau neuronal.
- **Extraction de caractéristiques** : Via *feature_extractor.py*, utilise MobileNetV3 et ses poids larges pour extraire des vecteurs caractéristiques.
- **Recherche de similarité** : Via *similarity_search.py*, la recherche se fait avec FAISS pour rechercher rapidement des images et la distance cosinus pour les catégories similaires.
- **Front-end** : Via *frontend.py*, utilise streamlit pour l'interface. 

## Architecture du Projet
```
projet/
│
├── setup.py                    # Fichier pour l'installation en mode éditable, générant le dossier projet.egg-info
├── requirements.txt            # Liste des dépendances
├── README.txt                  # Documentation du projet
│
├── src/
│   ├── __init__.py             # Marque le dossier comme un package Python
│   ├── image_preprocessing.py  # Module de prétraitement d'images
│   ├── feature_extractor.py    # Module d'extraction de caractéristiques
│   ├── similarity_search.py    # Module de recherche d'images similaires
│   ├── frontend.py             # Interface web via Streamlit
│
├── test/                       # Tests unitaires pour les modules
│   ├── image_preprocessing_test.py
│   ├── feature_extractor_test.py
│   ├── similarity_search_test.py
│   ├── app_test.py             # Test de l'ensemble des modules
│
├── ressources/                 # Dossier des embeddings, categories et images de Tiny ImageNet
│   ├── Tiny_ImageNet_Categories.npy      # Catégroies liées au images
│   ├── Tiny_ImageNet_Embeddings.npy      # Vecteurs de caractéristiques liées au images
│   ├── TinyImageNet_feature_extractor.py # Exctracteur des vecteurs du dataset
│   ├── tiny-imagenet-200                 # Dataset Tiny ImageNet, contenant 100k images
│
├── projet.egg-info/            # Métadonnées générées après l'installation en mode éditable 
│   ├── PKG-INFO                  # Contient les informations sur le package 
│   ├── requires.txt              # Liste des dépendances du package 
│   ├── top_level.txt             # Nom du package principal 
│   ├── dependency_links.txt      # Liens vers les dépendances externes 
│
```

## Prérequis
Voir **requirements.txt**. \
Concernant les ressources, le dossier est à télécharger depuis *[ce drive](https://drive.google.com/drive/folders/1dIx56IIORXPxI0vRue6CocAA-3QYxF0U?usp=sharing)* et à glisser dans le projet comme l'indique l'architecture.
## Tests Unitaires
Les tests unitaires se lancent via la commande (par exemple avec similarity_search_test.py) et depuis la racine du projet :
```
python3 -m unittest test/similarity_search_test.py
```

## Utilisation
Le lancement de l'**application web** se fait depuis la racine du projet avec la commande :
```
streamlit run src/frontend.py
```
Pour tester l'application **sans interface** et depuis la racine du projet :
```
python3 test/app_test.py
```
