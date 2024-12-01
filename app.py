import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Title of the Streamlit app
st.title("Face and Skin Type Detection")

# Sidebar description
st.sidebar.title("Upload an Image")
st.sidebar.write("Upload a clear face image for skin analysis.")

# File uploader widget
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load your TensorFlow model
@st.cache_resource
def load_model():
    # Load your pre-trained model here
    model = tf.keras.models.load_model("path_to_your_model.h5")  # Replace with your model's path
    return model

model = load_model()

# Function for processing the image
def preprocess_image(image):
    # Resize to model input size, normalize, etc.
    img_resized = cv2.resize(image, (224, 224))  # Replace (224, 224) with your model's input size
    img_normalized = img_resized / 255.0  # Normalize the image
    return np.expand_dims(img_normalized, axis=0)  # Add batch dimension

# Function for displaying results
def display_results(prediction):
    # Add logic for interpreting your model's output
    skin_types = ["Normal", "Oily", "Dry", "Combination"]  # Update with your actual classes
    skin_type = skin_types[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return skin_type, confidence

# Process the uploaded image
if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy array
    image_np = np.array(image)

    # Preprocess the image
    processed_image = preprocess_image(image_np)

    # Make predictions
    with st.spinner("Analyzing..."):
        prediction = model.predict(processed_image)

    # Get results
    skin_type, confidence = display_results(prediction)

    # Display results
    st.subheader("Analysis Results")
    st.write(f"**Detected Skin Type:** {skin_type}")
    st.write(f"**Confidence:** {confidence:.2f}%")
else:
    st.write("Please upload an image for analysis.")