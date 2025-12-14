import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd# ----------------------------------------------------
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# ----------------------------------------------------
#  LOAD MODEL & CSV
# ----------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mobilenetv2_insect.keras")  # your model file
    return model

@st.cache_data
def load_csv():
    expected_cols = [
    "Common Name", "Scientific Name", "Host Crops", "Damage Symptoms",
    "IPM Measures", "Chemical Control", "Kingdom", "Phylum", "Class",
    "Order", "Family", "Genus", "Species"
    ]
    df = pd.read_excel("insect species.xlsx")
    return df

model = load_model()
insect_df = load_csv()


# ----------------------------------------------------
#  PREDICTION FUNCTION
# ----------------------------------------------------
def predict_image(image):
    img = image.resize((160, 160))   # MobileNetV3 input size
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
    st.title("🐞INSECTIFICA🔍")
    st.write("""Insect Species Classification System""")
    st.subheader("Developed by Department of Biotechnology, St. Joseph's College (Autonomous), Trichy")

    st.write("""
        This AI-powered system identifies **insects & pests** up to genus–species level  
        providing reliable support for agricultural and biological studies..
        
        👉 Click **Next** to continue.
    """)

    if st.button("Next ➡️"):
        st.session_state.page = "about"


# ----------------------------------------------------
#  PAGE 2: ABOUT DEPARTMENT
# ----------------------------------------------------
def about_page():
    st.title("🏛️ About the Insectifica")
    st.write("""
**Insectifica** is an educational and research-support application developed by the  
**Department of Biotechnology, St. Joseph’s College (Autonomous), Trichy**.

The app is designed to serve as a reliable and user-friendly digital platform for understanding, identifying, and learning about a wide range of insect species commonly encountered in agricultural and ecological environments.

Insectifica integrates essential biological information, classification details, and reference data into a simple and accessible interface. Whether you are a student exploring entomology, a researcher conducting field studies, or an enthusiast seeking quick information, the app provides accurate insights to support learning needs.

With its structured layout, smooth navigation, and clear presentation of scientific data, Insectifica enhances academic engagement and makes insect identification more intuitive. It bridges traditional learning with modern digital tools, offering easy access to taxonomy, host plants, damage symptoms, and control measures—all in one place.

**Developed By:**  
Department of Biotechnology  
St. Joseph’s College (Autonomous), Trichy – 620002
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
        st.image(img, caption="Uploaded Insect Image", use_container_width=True)

        class_index, confidence = predict_image(img)

        row = insect_df.iloc[class_index]

        st.success(f"{row['Common Name']} ({row['Scientific Name']})")
       
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
