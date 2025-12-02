import streamlit as st
import tensorflow as tf
import numpy as np

# Load model once
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("SavedModel.h5")

model = load_cnn_model()

# Prediction Function
from PIL import Image

def model_prediction(test_image):
    image = Image.open(test_image)
    image = image.convert("RGB")
    image = image.resize((128, 128))
    input_arr = np.array(image).astype("float32")   # no normalization

    input_arr = np.expand_dims(input_arr, axis=0)
    return np.argmax(model.predict(input_arr))

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home.jpg", use_column_width=True)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
       🌱 Plant Disease Detection Web App

This project is a Plant Disease Detection System built using Deep Learning (TensorFlow) and deployed with an interactive Streamlit web interface.
It allows users to upload a plant leaf image, and the model predicts the type of disease (if any) based on visual symptoms.

🔍 Key Features

📸 Image Upload Support (JPG, PNG)

🤖 CNN-based Deep Learning Model for accurate disease classification

⚡ Real-time Prediction directly in the browser

🖼️ Displays uploaded image preview

🌿 Can be extended easily with more plant types or additional diseases

🛠️ Tech Stack

TensorFlow / Keras – Model building & prediction

Streamlit – Front-end web app

NumPy, Pillow – Image preprocessing

Python – Overall development

🧠 Model Info

The model is trained on plant leaf datasets with multiple classes of diseases and healthy leaves. It learns visual patterns such as spots, discoloration, blight, mildew, infection, etc.

🎯 Purpose

This tool aims to help farmers, students, and researchers quickly identify plant health issues using AI to promote better crop management and early disease detection.
                
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image:
        st.image(test_image, use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Please Wait..."):
            result_index = model_prediction(test_image)

            class_name =['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

            st.success(f"Model predicts: **{class_name[result_index]}**")
