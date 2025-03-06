import os
import requests
import tensorflow as tf
import numpy as np
from PIL import Image

# Chemin où le modèle sera sauvegardé localement
MODEL_PATH = os.path.join('models', 'cats_dogs_model.keras')
MODEL_URL = "https://drive.google.com/file/d/1_t6i9qqMEGm-aLO1oIgZw3dBzFjL1Vdr/view?usp=sharing"

# Cache pour le modèle chargé
_model = None

def download_model_if_needed():
    """Télécharge le modèle s'il n'existe pas déjà localement"""
    # Créer le dossier models s'il n'existe pas
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle...")
        
        try:
            import gdown
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        except ImportError:
                       
            # Enregistrer le fichier
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print(f"Modèle téléchargé et sauvegardé à {MODEL_PATH}")

def load_model():
    """Charge le modèle à partir du disque (avec mise en cache)"""
    global _model
    if _model is None:
        # Télécharger le modèle si nécessaire
        download_model_if_needed()
        
        # Charger le modèle
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def preprocess_image(image):
    """Prétraite une image pour la prédiction"""
    # Redimensionner l'image à la taille attendue par le modèle
    input_size = (500, 500)  # Ajustez selon votre modèle
    image = image.resize(input_size)
    
    # Convertir en tableau numpy et normaliser
    img_array = np.array(image) / 255.0
    
    # Ajouter la dimension du batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array