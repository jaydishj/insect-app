import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# ----------------------------------------------------
# MOBILE NET V3 PREPROCESSING
# ----------------------------------------------------
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# ----------------------------------------------------
#  LOAD MODEL & CSV
# ----------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mobilenetv3_insect_model.h5")  # your model file
    return model

@st.cache_data
def load_csv():
    df = pd.read_csv("insect_info.csv")
    return df

model = load_model()
insect_df = load_csv()


# ----------------------------------------------------
#  PREDICTION FUNCTION
# ----------------------------------------------------
def predict_image(image):
    img = image.resize((224, 224))   # MobileNetV3 input size
    img = np.array(img)

    img = preprocess_input(img)      # MobileNetV3 preprocessing
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return class_index, confidence


# ----------------------------------------------------
#  PAGE 1: WELCOME PAGE
# ----------------------------------------------------
def welcome_page():
    st.title("🪲 Insect Species Classification System")
    st.subheader("Developed by Department of Botany, St. Joseph's College (Autonomous), Trichy")

    st.write("""
        This AI-powered system identifies **insects & pests** up to genus–species level  
        using a MobileNetV3 deep learning model trained on **400 species**.
        
        👉 Click **Next** to continue.
    """)

    if st.button("Next ➡️"):
        st.session_state.page = "about"


# ----------------------------------------------------
#  PAGE 2: ABOUT DEPARTMENT
# ----------------------------------------------------
def about_page():
    st.title("🏛️ About the Department of Botany")
    st.write("""
        The Department of Botany at St. Joseph’s College, Trichy  
        is a leader in **plant sciences, biodiversity, ecology, and agriculture**.
        
        This project is developed as part of our **Digital Agriculture & AI** initiative.
    """)

    if st.button("Proceed to Classification ➡️"):
        st.session_state.page = "classification"


# ----------------------------------------------------
#  PAGE 3: CLASSIFICATION PAGE
# ----------------------------------------------------
def classification_page():
    st.title("🔍 Insect Image Classification")

    st.write("Upload an insect image to get species identification and management details.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Insect Image", use_column_width=True)

        class_index, confidence = predict_image(img)

        row = insect_df.iloc[class_index]

        st.success(f"**Predicted Species:** {row['Common Name']} ({row['Scientific Name']})")
        st.write(f"### Confidence: {confidence*100:.2f}%")

        st.write("## 🧬 Taxonomy")
        st.write(f"**Kingdom:** {row['Kingdom']}")
        st.write(f"**Phylum:** {row['Phylum']}")
        st.write(f"**Class:** {row['Class']}")
        st.write(f"**Order:** {row['Order']}")
        st.write(f"**Family:** {row['Family']}")
        st.write(f"**Genus:** {row['Genus']}")
        st.write(f"**Species:** {row['Species']}")

        st.write("## 🌿 Host Crops")
        st.write(row["Host Crops"])

        st.write("## 🐛 Damage Symptoms")
        st.write(row["Damage Symptoms"])

        st.write("## 🛡️ IPM Measures")
        st.write(row["IPM Measures"])

        st.write("## ⚠️ Chemical Control")
        st.write(row["Chemical Control"])

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "welcome"


# ----------------------------------------------------
#  STREAMLIT NAVIGATION LOGIC
# ----------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if st.session_state.page == "welcome":
    welcome_page()
elif st.session_state.page == "about":
    about_page()
elif st.session_state.page == "classification":
    classification_page()

st.write("---")
st.write("Thank you for using the AI-Driven Insect Classification System.")
