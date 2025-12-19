import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------------------------------------
# PAGE CONFIG (MUST BE FIRST)
# ----------------------------------------------------
st.set_page_config(
    page_title="INSECTIFICA | AI Insect Identification",
    page_icon="🐞",
    layout="centered"
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------
st.markdown("""
<style>
[data-testid="stToolbar"] {display: none !important;}

.stApp {
    background-color: #f5f7fa;
}

h1 {
    color: #1b4332;
    text-align: center;
    font-weight: 700;
}

h2, h3 {
    color: #2d6a4f;
}

p, li {
    font-size: 16px;
    line-height: 1.6;
}

div.stButton > button {
    background-color: #40916c;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-weight: 600;
    border: none;
}

div.stButton > button:hover {
    background-color: #2d6a4f;
}

hr {
    border: 1px solid #d8f3dc;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD MODEL & DATA
# ----------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenetv2_insect_best.keras")

@st.cache_data
def load_data():
    return pd.read_excel("insect species.xlsx")

model = load_model()
insect_df = load_data()

# ----------------------------------------------------
# PREDICTION FUNCTION
# ----------------------------------------------------
def predict_image(image):
    img = image.resize((160, 160))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    return np.argmax(preds), np.max(preds)

# ----------------------------------------------------
# SESSION STATE
# ----------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "intro"

# ----------------------------------------------------
# INTRO PAGE (IMAGE UPLOAD)
# ----------------------------------------------------
def intro_page():
    st.title("🐞 INSECTIFICA 🔍")
    st.subheader("AI-Powered Insect & Pest Identification")

    st.markdown("""
    **Insectifica** helps identify insects and pests instantly using artificial intelligence  
    and image recognition.

    Designed for **students, farmers, researchers, and nature enthusiasts**.
    """)

    st.divider()

    st.header("📸 Upload Insect Image")
    uploaded_file = st.file_uploader(
        "Capture or upload an insect image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.success("Image uploaded successfully")
        if st.button("🔍 Start Identification"):
            st.session_state.uploaded_image = img
            st.session_state.page = "classification"

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("ℹ️ About App"):
            st.session_state.page = "about_app"
    with col2:
        if st.button("👨‍🔬 Developers"):
            st.session_state.page = "developers"

# ----------------------------------------------------
# ABOUT APP PAGE
# ----------------------------------------------------
def about_app_page():
    st.title("ℹ️ About INSECTIFICA")

    st.markdown("""
    **Insectifica** is an AI-powered mobile application designed to help users instantly identify  
    insects, pests, and other arthropods from photographs.

    Developed as an **educational and research-support initiative** by the  
    **Department of Biotechnology, St. Joseph’s College (Autonomous), Tiruchirappalli**.
    """)

    st.divider()

    st.header("🎯 Core Purpose")
    st.markdown("""
    The primary goal of Insectifica is to provide **fast and accurate identification**  
    of insects and pests using smartphone images, along with educational insights.
    """)

    st.divider()

    if st.button("➡️ Features & Use Cases"):
        st.session_state.page = "features"

    if st.button("⬅️ Back"):
        st.session_state.page = "intro"

# ----------------------------------------------------
# FEATURES PAGE
# ----------------------------------------------------
def features_page():
    st.title("✨ Features & Use Cases")

    st.markdown("""
    • Instant AI-based insect identification  
    • Comprehensive species database  
    • Pest vs Beneficial classification  
    • Habitat & behaviour information  
    • Educational and research support
    """)

    st.divider()

    st.header("📸 Best Practices")
    st.markdown("""
    • Capture clear images  
    • Use good lighting  
    • Ensure wings, legs, and antennae are visible
    """)

    if st.button("👨‍🔬 Developers"):
        st.session_state.page = "developers"

    if st.button("⬅️ Back"):
        st.session_state.page = "about_app"

# ----------------------------------------------------
# DEVELOPERS PAGE
# ----------------------------------------------------
def developers_page():
    st.title("👨‍🔬 Development Team")

    st.markdown("""
    **Department of Biotechnology**  
    St. Joseph’s College (Autonomous)  
    Tiruchirappalli – 620 002
    """)

    st.divider()

    st.markdown("""
    **App Concept & Design**  
    Dr. A. Edward  

    **Development & Programming**  
    Dr. A. Edward  
    Dr. V. Swabna  
    Dr. A. Asha Monica  
    Dr. Pavulraj Michael  

    **Guidance & Supervision**  
    Dr. Pavulraj Michael SJ
    """)

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "intro"

# ----------------------------------------------------
# CLASSIFICATION PAGE
# ----------------------------------------------------
def classification_page():
    st.title("🔍 Insect Classification Result")

    img = st.session_state.get("uploaded_image", None)

    # 🔹 Case 1: No image uploaded
    if img is None:
        st.warning("No image uploaded.")
        if st.button("⬅️ Back"):
            st.session_state.page = "intro"
        return

    # 🔹 Case 2: Image exists → proceed
    st.image(img, use_container_width=True)

    with st.spinner("Analyzing insect image..."):
        class_index, confidence = predict_image(img)
        row = insect_df.iloc[class_index]

    st.success(
        f"{row['Common Name']} ({row['Scientific Name']})\n\n"
        f"Confidence: {confidence*100:.2f}%"
    )

    # 🧬 FULL TAXONOMY
    st.write("## 🧬 Taxonomy")
    st.write(f"**Kingdom:** {row['Kingdom']}")
    st.write(f"**Phylum:** {row['Phylum']}")
    st.write(f"**Class:** {row['Class']}")
    st.write(f"**Order:** {row['Order']}")
    st.write(f"**Family:** {row['Family']}")
    st.write(f"**Genus:** {row['Genus']}")
    st.write(f"**Species:** {row['Species']}")

    # 🌿 OTHER DETAILS
    st.write("## 🌿 Host Crops")
    st.write(row["Host Crops"])

    st.write("## 🐛 Damage Symptoms")
    st.write(row["Damage Symptoms"])

    st.write("## 🛡️ IPM Measures")
    st.write(row["IPM Measures"])

    st.write("## ⚠️ Chemical Control")
    st.write(row["Chemical Control"])


    if st.button("⬅️ Back to Home"):
        st.session_state.page = "intro"

# ----------------------------------------------------
# NAVIGATION
# ----------------------------------------------------
if st.session_state.page == "intro":
    intro_page()
elif st.session_state.page == "about_app":
    about_app_page()
elif st.session_state.page == "features":
    features_page()
elif st.session_state.page == "developers":
    developers_page()
elif st.session_state.page == "classification":
    classification_page()

st.write("---")
st.write("© Department of Biotechnology | St. Joseph’s College (Autonomous)")
