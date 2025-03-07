import streamlit as st
from PIL import Image
import numpy as np
from model_loader import load_model, preprocess_image
import os

# Configuration de la page
st.set_page_config(
    page_title="Classificateur Chien vs Chat",
    page_icon="🐱🐶",
    layout="centered"
)

# Fonction pour faire la prédiction
def predict(img):
    model = load_model()
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    return prediction[0][0]  # Pour un modèle de classification binaire

# Interface utilisateur
st.title("🐱 Classificateur Chien vs Chat 🐶")
st.write("Chargez une image de chien ou de chat pour obtenir une prédiction!")

# Upload d'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée", use_container_width=True)
    
    # Ajouter un bouton pour lancer la prédiction
    if st.button("Classifier l'image"):
        with st.spinner("Classification en cours..."):
            prediction = predict(image)
            
            # Afficher le résultat
            if prediction > 0.5:
                st.success(f"C'est un chien! (Confiance: {prediction:.2f})")
            else:
                st.success(f"C'est un chat! (Confiance: {1-prediction:.2f})")
            
            # Afficher la barre de progression
            chien_score = prediction
            chat_score = 1 - prediction
            
            st.write("Probabilité:")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Chat")
                st.progress(chat_score)
                st.write(f"{chat_score:.1%}")
            with col2:
                st.write("Chien")
                st.progress(float(chien_score))
                st.write(f"{chien_score:.1%}")