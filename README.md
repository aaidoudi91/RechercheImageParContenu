# Recherche d'image par contenu et indexation d'une base d'image
Ce projet implémente un système de recherche d'images similaires par le contenu *(CBIR - Content-based image retrieval)*, utilisant des descripteurs extraits de réseaux neuronaux convolutifs pour indexer et comparer les images de manière efficace. De plus, une fonctionnalité de recherche d'images similaires par le texte *(TBIR - Text-Based Image Retrieval)* a été implémentée, permettant d'effectuer des recherches d'images à partir de descriptions textuelles. 

## Modules du projet
- **Prétraitement des images** : Via *image_preprocessing.py*, le module prépare les images pour l'entrée dans un réseau neuronal.
- **Extraction de caractéristiques** : Via *feature_extractor.py*, utilise MobileNetV3 et ses poids larges pour extraire des vecteurs caractéristiques. Aussi, via *tinyimagenet_mobilenetv3_feature_extractor.py*, MobileNetV3 extrait les vecteurs du dataset.
- **Recherche de similarité** : Via *similarity_search.py*, la recherche se fait avec FAISS pour rechercher rapidement des images et la distance cosinus pour les catégories similaires.


- **Extraction de caractéristiques pour le TBIR** : Via *tinyimagenet_clip_feature_extractor.py*, utilise CLIP et son modèle ViT-B/32 pour extraire les vecteurs du dataset.
- **Recherche de similarité pour le TBIR** : Via *clip_similarity_search.py*, le module convertit le texte en un vecteur de caractéristiques pour le comparer à ceux du dataset.


- **Front-end** : Dans */frontend*, utilise Streamlit pour l'interface.

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
│   ├── clip_similarity_search.py  # Module de recherche d'images similaires via le texte
│   ├── frontend/               # Interface web via Streamlit
│
├── test/                       # Tests unitaires pour les modules
│   ├── image_preprocessing_test.py
│   ├── feature_extractor_test.py
│   ├── similarity_search_test.py
│   ├── app_test.py             # Test de l'ensemble des modules
│
├── ressources/                 # Dossier des embeddings, categories et images de Tiny ImageNet
│   ├── tiny-imagenet-200                 # Dataset Tiny ImageNet, contenant 100k images (non inclus - à télécharger)
│   ├── Tiny_ImageNet_MobilNetV3_Categories.npy       # Catégroies liées au images (pour le CBIR / non inclus - à télécharger)
│   ├── Tiny_ImageNet_MobilNetV3_Embeddings.npy       # Vecteurs de caractéristiques liées au images (pour le CBIR / non inclus - à télécharger)
│   ├── tinyimagenet_mobilnetv3_feature_extractor.py  # Exctracteur des vecteurs du dataset (pour le CBIR)
│   ├── Tiny_ImageNet_CLIP_Categories.npy       # Catégroies liées au images (pour le TBIR / non inclus - à télécharger)
│   ├── Tiny_ImageNet_CLIP_Embeddings.npy       # Vecteurs de caractéristiques liées au images (pour le TBIR / non inclus - à télécharger)
│   ├── tinyimagenet_clip_feature_extractor.py  # Exctracteur des vecteurs du dataset (pour le TBIR)
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
L'ensemble de données utilisé est **Tiny ImageNet**. Sous-ensemble d’ImageNet, il est conçu pour des expériences en classification d’images avec un dataset réduit contenant 200 classes de 500 images.\
**Source : [CS231N - Stanford](https://cs231n.stanford.edu/) - Téléchargeable depuis ce lien : [tiny-imagenet-200.zip](http://cs231n.stanford.edu/tiny-imagenet-200.zip).**

> **Important** : les fichiers d'embeddings et de catégories extraits via MobileNetV3 et CLIP sont disponibles sur *[ce drive](https://drive.google.com/drive/folders/1fG2j6oRhhP7w1kNZm0svfod8yZNfy3pU?usp=share_link)*.

### MobileNetV3 
Ce projet utilise **MobileNetV3**, un modèle de classification d'images développé par Google.  
**Source : [PyTorch MobileNetV3](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)**  

### CLIP - ViT-B/32
Ce projet utilise CLIP, un modèle multimodal développé par OpenAI capable d'associer des images et du texte pour une recherche et une classification avancées.
**Source : [OpenAI CLIP](https://openai.com/index/clip/)**

### FAISS
Ce projet utilise **Facebook AI Similarity Search** (FAISS) pour l'indexation et la recherche rapide de vecteurs d'images.\
**Source : [FAISS.IA](https://faiss.ai/)**

## Utilisation
Avant de pouvoir tester l'application, assurez-vous d'avoir téléchargé toutes les ressources nécessaires (dataset, embeddings et catégories) disponibles [ici](#tiny-imagenet).\
Ensuite, entrez la commande suivante pour télécharger les packages nécessaires : 
```
pip install -r requirements.txt
```

Le lancement de l'**application web** se fait avec la commande :
```
streamlit run src/frontend/main_frontend.py
```
Pour tester l'application **sans interface** :
```
python3 test/app_test.py <mobilenet|clip> <image_path|query_text>
```
