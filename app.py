import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# --------------------------------------------------
# Page Configuration (ONLY ONCE)
# --------------------------------------------------
import streamlit as st

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="INSECTIFICA",
    page_icon="🐞",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Custom CSS – GREEN TEXT EVERYWHERE
# --------------------------------------------------
st.markdown("""
<style>
/* Increase only main app title */
h1 {
    font-size: 45px !important;
    font-weight: 800;
    color: #1b5e20;
}

/* Mobile adjustment */
@media (max-width: 768px) {
    h1 {
        font-size: 36px !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ---------------- App Background ---------------- */
.stApp {
    background: linear-gradient(139deg, #f6fff8, #e8f5e9);
    color: #1b5e20;   /* Default text color */
}

/* ---------------- All Text Green ---------------- */
body, p, span, div, label {
    color: #1b5e20 !important;
}

/* ---------------- Titles ---------------- */
h1, h2, h3, h4, h5, h6 {
    text-align: center;
    color: #1b5e20 !important;
    font-weight: 700;
}

/* ---------------- Buttons ---------------- */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    background: linear-gradient(135deg, #2e7d32, #66bb6a);
    color: white !important;
    font-size: 18px;
    font-weight: bold;
    padding: 0.7em;
    border: none;
}

/* ---------------- Upload Box ---------------- */
[data-testid="stFileUploader"] {
    border: 2px dashed #2e7d32;
    border-radius: 16px;
    padding: 1em;
    background-color: #f1f8e9;
}

/* ---------------- File uploader text ---------------- */
[data-testid="stFileUploader"] * {
    color: #1b5e20 !important;
}

/* ---------------- Images ---------------- */
img {
    border-radius: 16px;
    max-width: 100%;
}

/* ---------------- Card Sections ---------------- */
.card {
    background: white;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    color: #1b5e20;
}

/* ---------------- Footer ---------------- */
.footer {
    text-align: center;
    font-size: 13px;
    color: #2e7d32 !important;
    margin-top: 30px;
}

/* =================================================
   📱 Mobile Responsive Styling
   ================================================= */
@media (max-width: 768px) {

    h1 { font-size: 26px; }
    h2 { font-size: 22px; }
    h3 { font-size: 18px; }

    .stButton > button {
        font-size: 16px;
        padding: 0.6em;
        border-radius: 12px;
    }

    .card {
        padding: 14px;
        border-radius: 14px;
    }

    .footer {
        font-size: 12px;
    }
}

</style>
""", unsafe_allow_html=True)




# ----------------------------------------------------
# LOAD MODEL & DATA
# ----------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mobilenetv2_insect_best.keras"
    )

# ----------------------------------------------------
#  LOAD JSON → DATAFRAME (FIXED)
# ----------------------------------------------------
@st.cache_data
def load_json():
    with open(
        "c:/Users/Admin/Downloads/full_insect_prediction_knowledge_base.json",
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
# SESSION STATE
# ----------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "intro"

# ----------------------------------------------------
# INTRO PAGE (IMAGE UPLOAD)
# ----------------------------------------------------
# ----------------------------------------------------

# Step 1: UI Card Function

def ui_card(title, content):
    st.markdown(
        f"""
        <div class="card">
            <h3>{title}</h3>
            <p>{content}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Step 2: How It Works Section
def how_it_works_section():
    ui_card(
        "🧠 How Insectifica Works",
        """
        <b>📸 Step 1: Snap or Upload a Photo</b><br>
        Use your device camera to take a clear, focused photo of the insect or pest,
        or upload an image from your gallery.<br><br>

        <b>🤖 Step 2: AI-Powered Analysis</b><br>
        Insectifica’s deep learning model analyzes the image by comparing it with a
        large entomological database, focusing on:
        <ul>
            <li>Body shape & size</li>
            <li>Color patterns</li>
            <li>Wing structure</li>
            <li>Antennae & leg features</li>
        </ul>

        <b>🐞 Step 3: Identification & Insights</b><br>
        Within seconds, the app provides:
        <ul>
            <li>Common & Scientific Name</li>
            <li>Taxonomic Classification</li>
            <li>Behaviour & Habitat</li>
            <li>Ecological Role (Pest / Beneficial / Neutral)</li>
        </ul>
        """
    )

    st.info(
        "💡 Tip: For best accuracy, ensure the insect is well-lit and clearly visible."
    )

# Optional: CSS for card styling
st.markdown("""
<style>
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.card h3 {
    color: #2d6a4f;
    margin-bottom: 10px;
}
.card p, .card li {
    font-size: 15px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)
def intro_page():
    st.title("🐞 INSECTIFICA 🔍")
    st.subheader("AI-Powered Insect & Pest Identification")

    st.markdown("""
    **Insectifica** helps identify insects and pests instantly using artificial intelligence  
    and image recognition.

    Designed for **students, farmers, researchers, and nature enthusiasts**.
    """)

    st.divider()

    # ✅ HOW IT WORKS SECTION (BEFORE UPLOAD)
    how_it_works_section()

    # ✅ IMAGE UPLOAD SECTION
    st.header("📸 Upload Insect / Pest Image")

    uploaded_file = st.file_uploader(
        "Capture or upload an insect image",
        type=["jpg", "jpeg", "png"],
        key="upload_image"
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.success("Image uploaded successfully")

        if st.button("🔍 Start Identification", key="start_identification"):
            st.session_state.uploaded_image = img
            st.session_state.page = "classification"


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

    st.markdown(
        """
        **Insectifica** is an AI-powered mobile application designed to help users instantly identify
        insects, pests, and other arthropods from photographs. It leverages advanced image recognition
        techniques and a comprehensive entomological database to make insect identification accessible
        to professionals, scientists, gardeners, farmers, and nature enthusiasts alike.

        Insectifica is an **educational and research-support application** developed by the  
        **Department of Biotechnology, St. Joseph’s College (Autonomous), Tiruchirappalli**.

        Developed with a commitment to educational and research excellence, Insectifica reflects
        St. Joseph’s College and the Department of Biotechnology’s ongoing mission to promote
        scientific awareness, support research, and create innovative tools that empower learners
        and professionals in the field of Biotechnology.
        """
    )

    st.divider()

    st.header("🎯 Core Purpose")
    st.markdown(
        """
        Insectifica’s primary goal is to provide **fast and accurate identification**
        of insects and pests using a simple photograph captured through a smartphone camera.

        Whether encountering a tiny beetle in a home garden, a mysterious insect indoors,
        or a potentially harmful pest in agricultural fields, Insectifica delivers
        **reliable identification results** along with **educational insights**—all with
        minimal effort.
        """
    )

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

    st.header("🔑 Key Features of Insectifica")
    st.markdown(
        """
        • **Instant Identification:**  
        Identify insects and arthropods instantly from photographs using advanced
        machine learning—ideal for both casual users and experts.

        • **Comprehensive Species Database:**  
        Access detailed profiles of hundreds of insect and pest species including
        butterflies, ants, beetles, moths, spiders, and major agricultural pests.

        • **Pest vs. Beneficial Indicator:**  
        Clearly distinguish whether a species is harmful (pest), neutral, or beneficial
        (such as pollinators and natural predators).

        • **Habitat & Behaviour Insights:**  
        Each identification includes habitat preferences, life cycle details, feeding
        habits, and ecological roles.

        • **Identification History:**  
        Save and review past identifications—useful for students, educators, researchers,
        and biodiversity documentation.

        • **Community & Sharing:**  
        Share discoveries with peers or within a community to encourage collaborative
        learning and nature awareness.
        """
    )

    st.divider()

    st.header("👥 Use Cases")
    st.markdown(
        """
        • **Gardeners & Homeowners:**  
        Identify plant pests and learn eco-friendly and sustainable management strategies.

        • **Students & Educators:**  
        Use real-world insect identifications for biology education, fieldwork, and projects.

        • **Farmers & Agriculturists:**  
        Detect agricultural pests early and make informed Integrated Pest Management (IPM)
        decisions.

        • **Nature Enthusiasts:**  
        Explore local biodiversity and maintain a personal record of insect sightings.
        """
    )

    st.divider()

    st.header("🌍 Why Insectifica Is Useful")
    st.markdown(
        """
        Insectifica bridges the gap between expert entomological knowledge and everyday
        curiosity. By combining artificial intelligence with scientifically curated
        databases, the application transforms insect encounters into meaningful educational
        experiences.

        It helps reduce fear and misinformation about insects while supporting biodiversity
        awareness, research documentation, and ecological understanding.
        """
    )

    st.divider()

    st.header("📸 Notes & Best Practices")
    st.markdown(
        """
        • Capture clear, well-focused images under good lighting conditions.  
        • Take photographs from multiple angles whenever possible.  
        • Ensure key anatomical features such as wings, legs, antennae, and body patterns
          are clearly visible to improve identification accuracy.
        """
    )

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
    Contact mail id:  
    edward_bt@mail.sjctni.edu  
    cisgene.edward@gmail.com
    """)

    st.divider()

    st.markdown("""
    **App Concept & Design**  
    Dr. A. Edward  

    **Development & Programming**  
    Dr. A. Edward  
    Dr. V. Swabna  
    Dr. A. Asha Monica  
    Dr. Pavulraj Michael SJ

    **Guidance & Supervision**  
    Dr. Pavulraj Michael SJ  
    Rector, St. Joseph’s College (Autonomous)  
    Tiruchirappalli – 620 002
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
    
    st.write("## 🌿 Host Crops")
    st.write(row["Host Crops"])

    st.write("## 🐛 Damage Symptoms")
    st.write(row["Damage Symptoms"])

    st.write("## 🛡️ IPM Measures")
    st.write(row["IPM Measures"])

    st.write("## ⚠️ Chemical Control")
    st.write(row["Chemical Control"])

    # 🌿 OTHER DETAILS

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
