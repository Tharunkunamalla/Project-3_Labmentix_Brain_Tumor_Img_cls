# Brain Tumor Classification Streamlit App

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load your trained model
MODEL_PATH = 'best_model.h5'  # Path to your trained model
model = load_model(MODEL_PATH)

# Define class labels (adjust based on your dataset)
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

def preprocess_image(img):
    """
    Preprocess the uploaded image to feed into the model
    """
    img = img.resize((224, 224))  # Resize to model input size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI scan image to predict the type of brain tumor.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

# After predicting
    with st.spinner('Classifying...'):
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)[0]
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")

    # --- Tumor Probability Graph ---
    st.subheader("Prediction Confidence for Each Class")

    fig, ax = plt.subplots()
    colors = ['green' if i == np.argmax(prediction) else 'skyblue' for i in range(len(CLASS_NAMES))]
    bars = ax.bar(CLASS_NAMES, prediction, color=colors)

    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Confidence for Each Tumor Type")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center', fontsize=9)
    st.pyplot(fig)


st.markdown("""
---
**Note**: This tool is for educational purposes and should not be used as a substitute for professional medical diagnosis.
""")
