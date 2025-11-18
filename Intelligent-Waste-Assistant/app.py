import os
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="🌱",
    layout="centered"
)

# ==========================================
# 2. SIDEBAR (ABOUT SECTION)
# ==========================================
with st.sidebar:
    st.title("ℹ️ About")
    st.info(
        "This project utilizes Transfer Learning with "
        "MobileNetV2 to classify waste into 4 categories. "
        "Trained on the Kaggle Garbage Classification dataset."
    )
    st.markdown("---")
    st.write("**Developed by:** Astha Priyam(●'◡'●)")
    st.write("**Version:** 1.0.0")

# ==========================================
# 3. MODEL LOADING (FIXED PATH LOGIC)
# ==========================================
@st.cache_resource
def load_model():
    # Get the absolute path to the current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to the model file
    model_path = os.path.join(curr_dir, 'final_model.h5')
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    return model

# Attempt to load the model with a spinner
try:
    with st.spinner('Loading AI Model...'):
        model = load_model()
except Exception as e:
    st.error(f"Error loading model. Please ensure 'final_model.h5' is in the repository. Details: {e}")

# ==========================================
# 4. CLASS NAMES
# ==========================================
# These must match the alphabetical order of your training folders
class_names = ['E-waste_Hazardous', 'Landfill_General', 'Organic_Compostable', 'Recyclables']

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
def import_and_predict(image_data, model):
    # Resize image to 224x224 (MobileNetV2 standard)
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img = np.asarray(image)
    img = img / 255.0
    
    # Reshape for the model (1, 224, 224, 3)
    img_reshape = img[np.newaxis, ...]
    
    # Predict
    prediction = model.predict(img_reshape)
    return prediction

# ==========================================
# 6. MAIN UI
# ==========================================
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🌱 EcoSort: Intelligent Waste Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image of waste to classify it instantly.</p>", unsafe_allow_html=True)

file = st.file_uploader("Choose a waste image...", type=["jpg", "png", "jpeg"])

if file is None:
    st.info("Please upload an image file to start.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    if st.button("Classify Waste"):
        try:
            predictions = import_and_predict(image, model)
            score = tf.nn.softmax(predictions[0])
            class_index = np.argmax(predictions[0])
            result = class_names[class_index]
            confidence = 100 * np.max(score)
            
            st.markdown("---")
            st.subheader(f"Prediction: {result}")
            st.caption(f"Confidence: {confidence:.2f}%")
            
            # Display Custom Action Messages
            if result == 'Recyclables':
                st.success("♻️ **Action:** Rinse and place in the **BLUE** bin. Ensure it is clean.")
            elif result == 'Organic_Compostable':
                st.warning("🍂 **Action:** Place in the **GREEN** bin or your compost pile.")
            elif result == 'E-waste_Hazardous':
                st.error("🔋 **Action:** Do NOT trash. Take to a designated **E-Waste Collection Point**.")
            else:
                st.info("🗑️ **Action:** Place in the **BLACK/GREY** bin (General Landfill).")
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")