import os
import requests
import tensorflow as tf
import numpy as np
from PIL import Image

# Chemin où le modèle sera sauvegardé localement
MODEL_PATH = os.path.join('models', 'cats_dogs_model.keras')
# URL S3 de votre modèle
MODEL_URL = "https://msde1.s3.eu-north-1.amazonaws.com/Cat%26Dogs_model_final.keras"

# Cache pour le modèle chargé
_model = None

def download_model_if_needed():
    """Télécharge le modèle depuis AWS S3 s'il n'existe pas déjà localement"""
    # Créer le dossier models s'il n'existe pas
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle depuis AWS S3...")
        
        try:
            # Téléchargement direct avec requests
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            # Enregistrer le fichier en écrivant par blocs
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            print(f"Modèle téléchargé et sauvegardé à {MODEL_PATH}")
        except Exception as e:
            print(f"Erreur lors du téléchargement du modèle: {e}")
            raise

def load_model():
    """Charge le modèle à partir du disque (avec mise en cache)"""
    global _model
    if _model is None:
        # Télécharger le modèle si nécessaire
        download_model_if_needed()
        
        # Charger le modèle
        try:
            _model = tf.keras.models.load_model(MODEL_PATH)
            print("Modèle chargé avec succès!")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            raise
            
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