# Recherche d'image par contenu et indexation d'une base d'image
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
│   ├── frontend/               # Interface web via Streamlit
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
│   ├── PKG-INFO                # Contient les informations sur le package 
│   ├── requires.txt            # Liste des dépendances du package 
│   ├── top_level.txt           # Nom du package principal 
│   ├── dependency_links.txt    # Liens vers les dépendances externes 
│
```

## Ressources
### Tiny ImageNet
L'ensemble de données utilisé est **Tiny ImageNet**. Sous-ensemble d’ImageNet, il est conçu pour des expériences en classification d’images avec un dataset réduit contenant 200 classes de 500 images (64×64).\
**Source : [CS231N - Stanford](https://cs231n.stanford.edu/) - fourni pour un usage académique et non commercial.**

> Le dossier des ressources, en son intégralité (dataset, embeddings et catégories), est disponible au besoin depuis *[ce drive](https://drive.google.com/drive/folders/1dIx56IIORXPxI0vRue6CocAA-3QYxF0U?usp=sharing)*.

### MobileNetV3 
Ce projet utilise **MobileNetV3**, un modèle de classification d'images développé par Google.  
Le modèle et ses poids sont distribués sous licence [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).\
**Source : [PyTorch MobileNetV3](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)**  

### FAISS

Ce projet utilise **Facebook AI Similarity Search** (FAISS) pour l'indexation et la recherche rapide de vecteurs d'images. Développé par Facebook AI Research, FAISS est distribué sous [licence MIT](https://github.com/facebookresearch/faiss/blob/main/LICENSE).\
**Source : [FAISS.IA](https://faiss.ai/)**

## Utilisation
Le lancement de l'**application web** se fait avec la commande :
```
streamlit run src/frontend.py
```
Pour tester l'application **sans interface** :
```
python3 test/app_test.py <chemin/vers/image.jpg>
```
