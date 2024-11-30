import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('1.h5')
    return model

# Preprocessing and prediction function
def model_prediction(uploaded_image):
    # Load the uploaded image and preprocess it
    img = tf.keras.preprocessing.image.load_img(uploaded_image, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = tf.expand_dims(input_arr, axis=0)  # Add batch dimension
    input_arr = input_arr / 255.0  # Normalize pixel values
    
    # Predict using the model
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Class names (should match your training dataset)
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Streamlit app
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select page", ["Home", "About", "Disease Recognition"])

if app_mode == 'Home':
    st.header("POTATO DISEASE RECOGNITION SYSTEM")
    st.image('home_page.jpg', use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    **How It Works:**
    1. Upload an image on the **Disease Recognition** page.
    2. Our model analyzes the image and predicts the disease.
    3. Get actionable insights for protecting crops.

    **Why Choose Us:**
    - Accurate predictions with state-of-the-art deep learning.
    - Intuitive interface for easy navigation.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ### Dataset Summary:
    - **Images:** 1558 total
    - **Classes:** Potato__Early_blight, Potato__Late_blight, Potato__Healthy
    - **Image Size:** 256x256 pixels
    - **Accuracy:** Training 98.77%, Validation 99.87%

    ### Objective:
    Help farmers detect potato leaf diseases early and ensure better crop yield through modern technology.
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    uploaded_image = st.file_uploader("Upload an Image")
    
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        
    if st.button("Predict"):
        with st.spinner("Analysing:"):
                st.write("Analyzed...")
                model = load_model()
                result_index = model_prediction(uploaded_image)
                predicted_class = class_names[result_index]
                st.success(f"The model predicts: **{predicted_class}**")
            
            
