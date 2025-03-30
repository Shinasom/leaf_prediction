import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Define model path
model_path = 'trained_plant_disease_model.keras'

# Google Drive file ID for downloading the model
file_id = "1--SDtd_id0vlY6aRS0sSUnZf5x7lqXOE"
url = f"https://drive.google.com/uc?id={file_id}"

# Download the model if it does not exist
if not os.path.exists(model_path):
    st.warning("Downloading the model file. Please wait...")
    try:
        gdown.download(url, model_path, quiet=False)
        if not os.path.exists(model_path):
            st.error("Model download failed. Please check the file ID.")
    except Exception as e:
        st.error(f"Error downloading model: {e}")

# Load the trained model
try:
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully! ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Prevent errors if the model is not loaded

# Function to predict the disease from an image
def model_prediction(test_image):
    try:
        # Open image and preprocess
        image = Image.open(test_image).convert("RGB")
        image = image.resize((128, 128))  # Resize to match model input shape
        input_arr = np.array(image) / 255.0  # Normalize pixel values
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # Return the predicted class index
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Streamlit UI
st.sidebar.title("🌱 PLANT DISEASE DETECTION SYSTEM")
app_mode = st.sidebar.selectbox("Select page", ["🏠 Home", "🔍 Disease Recognition"])

if app_mode == "🏠 Home":
    st.markdown("<h1 style='text-align: center; color: red;'>PLANT DISEASE DETECTION SYSTEM</h1>", unsafe_allow_html=True)
    st.write("### 📌 Upload an image to detect plant diseases using AI!")

elif app_mode == "🔍 Disease Recognition":
    st.header('🌿 Plant Disease Detection')
    test_image = st.file_uploader('📷 Upload an image:', type=['jpg', 'png', 'jpeg'])

    if test_image:
        st.image(test_image, use_column_width=True)

        if st.button("🔎 Predict"):
            st.snow()  # Add cool visual effect
            if model is None:
                st.error("⚠ Model is not loaded. Please check for errors.")
            else:
                result_index = model_prediction(test_image)

                if result_index is not None:
                    # Define class names
                    class_names = ['Potato Early Blight', 'Potato Late Blight', 'Healthy Potato']
                    st.success(f"✅ Model Prediction: **{class_names[result_index]}**")
