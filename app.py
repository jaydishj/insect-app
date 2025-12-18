import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd# ----------------------------------------------------
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------------------------------------
#  LOAD MODEL & CSV
# ----------------------------------------------------
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
st.set_page_config(
    page_title="Insectifica | Insect & Pest Identification",
    page_icon="🐞",
    layout="centered"
)

load_custom_css()   
st.markdown("""
<style>
[data-testid="stToolbar"] {display: none !important;}
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

def welcome_page():
    st.title("🐞 INSECTIFICA")
    st.subheader("Insect and Pest Identification Mobile Application")

    st.markdown(
        """
        **Insectifica** is an AI-powered mobile application designed to help users instantly identify  
        insects, pests, and other arthropods from photographs. The application leverages advanced  
        image recognition techniques and a comprehensive entomological database to make insect  
        identification accessible to professionals, scientists, gardeners, farmers, students, and  
        nature enthusiasts.

        Developed as an **educational and research-support initiative** by the  
        **Department of Biotechnology, St. Joseph’s College (Autonomous), Tiruchirappalli**,  
        Insectifica reflects the institution’s commitment to scientific excellence, innovation,  
        and community-oriented learning.
        """
    )

    st.divider()

    st.header("🎯 Core Purpose")
    st.markdown(
        """
        The primary goal of Insectifica is to provide **fast and accurate identification** of insects  
        and pests using a simple photograph captured through a smartphone camera.  

        Whether encountering an unfamiliar insect in a home environment, a garden, or an  
        agricultural field, Insectifica delivers **reliable identification results** along with  
        **educational insights**, enabling informed decision-making and learning.
        """
    )

    st.divider()

    st.header("⚙️ How It Works")
    st.markdown(
        """
        **1. Capture or Upload an Image**  
        Use the in-app camera to take a clear photograph of the insect or upload an existing image  
        from the device gallery.

        **2. AI-Based Image Analysis**  
        The AI model analyzes the image by comparing it with a trained database of insect and pest  
        species, focusing on visual traits such as body structure, coloration, wings, and antennae.

        **3. Identification & Information Output**  
        Within seconds, the application provides the most probable species identification along with  
        scientific details including taxonomy, common name, habitat, behaviour, and ecological role.
        """
    )

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
