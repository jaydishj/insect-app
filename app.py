import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------------------------------------
# PAGE STYLE
# ----------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: transparent;
    background-image:
        linear-gradient(135deg, rgba(0,0,0,0.04) 25%, transparent 25%),
        linear-gradient(225deg, rgba(0,0,0,0.04) 25%, transparent 25%),
        linear-gradient(315deg, rgba(0,0,0,0.04) 25%, transparent 25%),
        linear-gradient(45deg,  rgba(0,0,0,0.04) 25%, transparent 25%);
    background-size: 40px 40px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mobilenetv2_insect_best.keras"
    )

# ----------------------------------------------------
# LOAD JSON → DATAFRAME
# ----------------------------------------------------
@st.cache_data
def load_json():
    with open(
        "pest.json",
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    rows = []
    for _, item in data.items():
        tax = item.get("taxonomy", {})
        rows.append({
            "Common Name": item.get("common_name", ""),
            "Scientific Name": item.get("scientific_name", ""),
            "Host Crops": ", ".join(item.get("host_crops", [])),
            "Damage Symptoms": ", ".join(item.get("damage_symptoms", [])),
            "IPM Measures": ", ".join(item.get("ipm_measures", [])),
            "Chemical Control": ", ".join(item.get("chemical_control", [])),
            "Kingdom": tax.get("kingdom", ""),
            "Phylum": tax.get("phylum", ""),
            "Class": tax.get("class", ""),
            "Order": tax.get("order", ""),
            "Family": tax.get("family", ""),
            "Genus": tax.get("genus", ""),
            "Species": tax.get("species", "")
        })

    return pd.DataFrame(rows)

model = load_model()
insect_df = load_json()

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
# PAGE 1 – HOME
# ----------------------------------------------------
def home_page():
    st.title("🐞 INSECTIFICA 🔍")
    st.subheader("AI-Based Insect Species Classification")
    st.write("Upload an insect image to identify species and control measures.")

    uploaded_file = st.file_uploader(
        "Upload insect image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.session_state.image = img
        st.session_state.page = "result"

# ----------------------------------------------------
# PAGE 2 – RESULT
# ----------------------------------------------------
def result_page():
    st.title("🔍 Classification Result")

    img = st.session_state.get("image")

    if img is None:
        st.error("No image found.")
        st.session_state.page = "home"
        return

    st.image(img, use_container_width=True)

    with st.spinner("Analyzing image..."):
        class_index, confidence = predict_image(img)

    if insect_df.empty or class_index >= len(insect_df):
        st.error("Prediction index out of range.")
        return

    row = insect_df.iloc[class_index]

    st.success(
        f"{row['Common Name']} ({row['Scientific Name']})"
        f"\n\nConfidence: {confidence*100:.2f}%"
    )

    st.subheader("🧬 Taxonomy")
    for k in ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]:
        st.write(f"**{k}:** {row[k]}")

    st.subheader("🌿 Host Crops")
    st.write(row["Host Crops"])

    st.subheader("🐛 Damage Symptoms")
    st.write(row["Damage Symptoms"])

    st.subheader("🛡️ IPM Measures")
    st.write(row["IPM Measures"])

    st.subheader("⚠️ Chemical Control")
    st.write(row["Chemical Control"])

    if st.button("⬅️ Back"):
        st.session_state.page = "home"

# ----------------------------------------------------
# STREAMLIT ROUTER
# ----------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "result":
    result_page()

st.write("---")
st.caption("Department of Biotechnology – St. Joseph’s College (Autonomous), Trichy")
