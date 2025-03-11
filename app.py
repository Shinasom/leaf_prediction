import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Define model path
model_path = 'trained_plant_disease_model.keras'

# Google Drive file ID for downloading the model (replace with actual file ID)
file_id = "YOUR_FILE_ID"
url = f"https://drive.google.com/uc?id={file_id}"

# Download the model if it does not exist
if not os.path.exists(model_path):
    st.warning("Downloading the model file. Please wait...")
    gdown.download(url, model_path, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Function to predict the disease from an image
def model_prediction(test_image):
    # Open image and preprocess
    image = Image.open(test_image)
    image = image.resize((128, 128))  # Resize to match model input shape
    input_arr = np.array(image) / 255.0  # Normalize pixel values
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return the predicted class index

# Streamlit UI
st.sidebar.title("PLANT DISEASE DETECTION SYSTEM")
app_mode = st.sidebar.selectbox("Select page", ["Home", "DISEASE RECOGNITION"])

if app_mode == "Home":
    st.markdown("<h1 style='text-align: center; color: red;'>PLANT DISEASE DETECTION SYSTEM</h1>", unsafe_allow_html=True)

elif app_mode == "DISEASE RECOGNITION":
    st.header('Plant Disease Detection')
    test_image = st.file_uploader('Choose an image:', type=['jpg', 'png', 'jpeg'])

    if test_image:
        st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            result_index = model_prediction(test_image)

            # Define class names
            class_names = ['Potato Early Blight', 'Potato Late Blight', 'Healthy Potato']
            st.success(f"Model is predicting: {class_names[result_index]}")
