import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd# ----------------------------------------------------
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------------------------------------
#  LOAD MODEL & CSV
# ----------------------------------------------------
st.markdown("""
<style>
[data-testid="stToolbar"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

def load_custom_css():
    st.markdown(
        """
        <style>
        /* Main background */
        .stApp {
            background-color: #f5f7fa;
        }

        /* Title styling */
        h1 {
            color: #1b4332;
            text-align: center;
            font-weight: 700;
        }

        /* Header styling */
        h2, h3 {
            color: #2d6a4f;
        }

        /* Markdown text */
        p, li {
            font-size: 16px;
            line-height: 1.6;
        }

        /* Buttons */
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
            color: #ffffff;
        }

        /* Divider */
        hr {
            border: 1px solid #d8f3dc;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

load_custom_css()   
st.markdown("""
<style>
[data-testid="stToolbar"] {display: none !important;}
</style>
""", unsafe_allow_html=True)
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
    background-position: 20px 0, 20px 0, 0 0, 0 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mobilenetv2_insect_best.keras")  # your model file
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

# Page configuration (MUST be first Streamlit command)
st.set_page_config(
    page_title="Insectifica | Insect & Pest Identification",
    page_icon="🐞",
    layout="centered"
)
# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "intro"

# Page routing
if st.session_state.page == "intro":
    intro_page()
elif st.session_state.page == "about":
    about_page()

def intro_page():
    st.title("🐞 INSECTIFICA 🔍")
    st.subheader("Insect & Pest Identification App")

    st.markdown(
        """
        **Insectifica** is an AI-powered mobile application that helps users  
        instantly identify insects and pests using photographs.

        Designed for **students, farmers, researchers, and nature enthusiasts**,  
        the app makes insect identification simple, fast, and educational.
        """
    )

    st.divider()

    st.header("📸 Upload Insect Image")
    uploaded_file = st.file_uploader(
        "Capture or upload an insect image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.success("Image uploaded successfully!")

    st.divider()

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Next ➡️"):
            st.session_state.page = "about"

    st.divider()
    if st.button("Next ➡️"):
        st.session_state.page = "about"


# ----------------------------------------------------
#  PAGE 2: ABOUT DEPARTMENT
# ----------------------------------------------------
def about_page():
    st.title("🏛️ About Insectifica")

    st.header("✨ Key Features")
    st.markdown(
        """
        • **Instant Identification:** Rapid insect and arthropod identification using advanced machine learning.  
        • **Comprehensive Species Database:** Detailed profiles of butterflies, ants, beetles, moths, spiders,  
          and agricultural pests.  
        • **Pest vs. Beneficial Indicator:** Clear classification of species as harmful, neutral, or beneficial  
          (pollinators or natural predators).  
        • **Habitat & Behaviour Insights:** Educational content on life cycles, feeding habits, and ecological roles.  
        • **Identification History:** Personal log of past identifications for academic, research, and reference use.  
        • **Community & Sharing:** Share discoveries and collaborate through a nature-focused user community.
        """
    )

    st.divider()

    st.header("👥 Use Cases")
    st.markdown(
        """
        • **Gardeners & Homeowners:** Identify pests and learn eco-friendly and sustainable management strategies.  
        • **Students & Educators:** Support biology education through real-world field identification activities.  
        • **Farmers & Agriculturists:** Enable early detection of agricultural pests and informed pest management decisions.  
        • **Nature Enthusiasts:** Explore local biodiversity and maintain personal insect sighting records.
        """
    )

    st.divider()

    st.header("🌍 Why Insectifica Is Useful")
    st.markdown(
        """
        Insectifica bridges the gap between expert entomological knowledge and everyday curiosity.  
        By integrating artificial intelligence with scientifically verified data, the application  
        transforms insect encounters into educational experiences, reduces misinformation, and  
        supports biodiversity awareness and ecological research.
        """
    )

    st.divider()

    st.header("📸 Notes & Best Practices")
    st.markdown(
        """
        • Capture clear, well-focused images under good lighting conditions.  
        • Take photographs from multiple angles whenever possible.  
        • Ensure key anatomical features such as wings, legs, antennae, and body patterns are visible  
          for improved identification accuracy.
        """
    )

    st.divider()

    st.header("🏫 Developed By")
    st.markdown(
        """
        **Department of Biotechnology**  
        St. Joseph’s College (Autonomous)  
        Tiruchirappalli – 620 002
        """
    )

    st.header("👨‍🔬 Project Team")
    st.markdown(
        """
        **App Concept & Design:**  
        Dr. A. Edward  

        **Development & Programming:**  
        Dr. A. Edward  
        Dr. V. Swabna  
        Dr. A. Asha Monica  
        Dr. Pavulraj Michael  

        **Scientific Data Verification:**  
        Dr. V. Swabna  
        Dr. A. Asha Monica  
        Dr. Pavulraj Michael  

        **Guidance & Supervision:**  
        Dr. Pavulraj Michael SJ  
        Rector , St. Joseph’s College (Autonomous)   
        Tiruchirappalli – 620 002
        """
    )

    st.divider()

    st.header("📬 Contact")
    st.markdown(
        """
        **Department of Biotechnology**  
        St. Joseph’s College (Autonomous)  
        Tiruchirappalli – 620 002
        """
    )

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
