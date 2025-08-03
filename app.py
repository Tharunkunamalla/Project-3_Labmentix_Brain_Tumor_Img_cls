# Brain Tumor Classification Streamlit App

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ================= Build exact model architecture =================
def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None  # Important: Use None because we are loading our own weights
    )

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    predictions = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    return model

# ================= Load model and weights =================
MODEL_PATH = 'best_model.h5'

try:
    model = build_model()
    model.load_weights(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ================= Class labels =================
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# ================= Preprocessing =================
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ================= Streamlit UI =================
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI scan image to predict the type of brain tumor.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    with st.spinner('🔍 Classifying...'):
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)[0]
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

    st.success(f"🧾 **Prediction:** {predicted_class}")
    st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")

    # --- Tumor Probability Graph ---
    st.subheader("📈 Prediction Confidence for Each Class")

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
📝 **Note**: This tool is for educational purposes and should not replace professional medical diagnosis.
""")
