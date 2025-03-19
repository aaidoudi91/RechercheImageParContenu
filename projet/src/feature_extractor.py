""" Module de feature extraction avec MobileNetV3
Ce module permet d'extraire le vecteur de caractéristiques d'une image à l'aide du modèle
MobileNetV3 pré-entraîné sur ImageNet (avec les poids large). Le module intègre le prétraitement de l'image,
soit à travers un module personnalisé (s'il est disponible), soit via un pipeline de transformations de secours.
Le vecteur retourné correspond aux embeddings issus du passage de l'image par le réseau, après
un pooling global et un aplatissement. """

import logging, torch, numpy as np
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

try: # Tenter d'importer le module de prétraitement personnalisé
    from image_preprocessing import preprocess_image
except ImportError:
    preprocess_image = None
    logging.warning("Module de prétraitement d'image non trouvé. Utilisation d'un prétraitement minimal.")

class FeatureExtractor:
    """ Extrait les caractéristiques d'une image avec MobileNetV3. """

    def __init__(self, device: str = None, target_size: tuple[int, int] = (224, 224)):
        """ Initialise MobileNetV3 en mode évaluation et prépare le pipeline d'extraction des features.
        :param device: 'cuda' ou 'cpu'. Si None, l'appareil est déduit automatiquement.
        :param target_size: Dimension d'entrée exigée par le modèle (largeur, hauteur). """
        if device is None: # Détermination automatique du device si non spécifié
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 # Chargement du modèle MobileNetV3 pré-entraîné sur ImageNet
        self.model = mobilenet_v3_large(weights=weights)
        self.model.eval() # Passage en mode évaluation (désactive dropout, etc.)
        self.model.to(self.device) # Déplacement du modèle sur le device approprié
        self.target_size = target_size

        # Retire la partie classification via l'utilisation des couches features et avgpool
        self.feature_extractor = torch.nn.Sequential(self.model.features, self.model.avgpool, torch.nn.Flatten())

        # Pipeline de prétraitement minimal en cas d'absence du module personnalisé
        self.basic_transform = Compose([
            Resize(self.target_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],  # Valeurs de normalisation standard pour ImageNet
                      std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_input, from_preprocessed: bool = False) -> np.ndarray:
        """ Extrait et retourne le vecteur de caractéristiques de l'image.
        :param image_input: Chemin vers l'image ou objet PIL.Image.
        :param from_preprocessed: Si True, l'image est déjà sous forme de tensor.
        :return: Vecteur 1D des caractéristiques en format numpy. """

        if from_preprocessed:
            image_tensor = image_input
        else:
            if preprocess_image: # Utilisation du module personnalisé de prétraitement
                image_tensor = preprocess_image(image_input, target_size=self.target_size, to_tensor=True)
            else: # Prétraitement minimal en cas d'absence du module dédié
                try:
                    image = Image.open(image_input).convert("RGB")
                except Exception as e:
                    raise ValueError(f"Erreur lors de l'ouverture de l'image {image_input}: {e}")
                image_tensor = self.basic_transform(image)

        if image_tensor.ndim == 3: # Ajoute une dimension batch si nécessaire
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad(): # Désactive le calcul des gradients pour l'inférence
            features = self.feature_extractor(image_tensor)
        return features.cpu().numpy().flatten() # Retourne le vecteur de caractéristiques sous forme de tableau numpy