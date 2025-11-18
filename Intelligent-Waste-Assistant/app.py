import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="EcoSort AI", page_icon="🌱")

# --- THE SIDEBAR CODE  ---
with st.sidebar:
    st.title("ℹ️ About")
    st.info(
        "This project utilizes Transfer Learning with MobileNetV2 "
        "to classify waste into 4 categories. Trained on the "
        "Kaggle Garbage Classification dataset."
    )
    st.markdown("---")
    st.write("**Developed by:** Astha Priyam(●'◡'●)")
    st.write("**Version:** 1.0.0")
# ------------------------------------

# 2. Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

# --- HIDE WARNINGS ---
st.set_option('deprecation.showfileUploaderEncoding', False)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('waste_classifier.h5')
    return model

with st.spinner('Loading AI Model...'):
    model = load_model()

# --- CLASS NAMES (MUST MATCH COLAB OUTPUT) ---
class_names = ['E-waste_Hazardous', 'Landfill_General', 'Organic_Compostable', 'Recyclables']

# --- UI ---
st.title("🌱 EcoSort: Intelligent Waste Assistant")
st.write("Upload an image of waste to classify it instantly.")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img = img / 255.0
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.info("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    if st.button("Analyze Waste"):
        predictions = import_and_predict(image, model)
        score = tf.nn.softmax(predictions[0])
        class_index = np.argmax(predictions[0])
        result = class_names[class_index]
        confidence = 100 * np.max(score)
        
        st.markdown("---")
        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence: {confidence:.2f}%")
        
        # Actionable Advice
        if result == 'Recyclables':
            st.success("♻️ **RECYCLE:** Rinse and place in the recycling bin.")
        elif result == 'Organic_Compostable':
            st.warning("🍂 **COMPOST:** Place in your compost bin or green waste.")
        elif result == 'E-waste_Hazardous':
            st.error("🔋 **HAZARDOUS:** Do NOT throw in trash. Take to an e-waste center.")
        else:
            st.info("🗑️ **LANDFILL:** This item goes in the general trash bin.")