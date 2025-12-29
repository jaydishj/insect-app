import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

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
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(139deg, #f6fff8, #e8f5e9);
        color: #1b5e20;
    }
    h1, h2, h3, h4, h5, h6 {
        text-align: center;
        color: #1b5e20 !important;
        font-weight: 700;
    }
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
    [data-testid="stFileUploader"] {
        border: 2px dashed #2e7d32;
        border-radius: 16px;
        padding: 1em;
        background-color: #f1f8e9;
    }
    img {
        border-radius: 16px;
        max-width: 100%;
    }
    .card {
        background: white;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        color: #1b5e20;
    }
    .footer {
        text-align: center;
        font-size: 13px;
        color: #2e7d32 !important;
        margin-top: 30px;
    }
    @media (max-width: 768px) {
        h1 { font-size: 36px !important; }
        .stButton > button { font-size: 16px; padding: 0.6em; }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Data & Model
# --------------------------------------------------
with open("pest.json", "r") as f:
    insect_data = json.load(f)

class_names = [
    'Acanthophilus helianthi rossi', 'Achaea janata', 'Acherontia styx', 'Adisura atkinsoni',
    'Aedes aegypti', 'Aedes albopictus', 'Agrotis ipsilon', 'Alcidodes affaber',
    'Aleurodicus dispersus', 'Amsacta albistriga', 'Anarsia ephippias', 'Anarsia epoitas',
    'Anisolabis stallii', 'Antestia cruciata', 'Aphis craccivora', 'Apis mellifera',
    'Apriona cinerea', 'Araecerus fasciculatus', 'Atractomorpha crenulata', 'Autographa nigrisigna',
    'Bagrada hilaris', 'Basilepta fulvicorne', 'Batocera rufomaculata', 'Calathus erratus',
    'Camponotus consobrinus', 'Chilasa clytia', 'Chilo sacchariphagus indicus',
    'Conogethes punctiferalis', 'Danaus plexippus', 'Dendurus coarctatus',
    'Deudorix (Virachola) isocrates', 'Elasmopalpus jasminophagus', 'Euwallacea fornicatus',
    'Ferrisia virgata', 'Formosina flavipes', 'Gangara thyrsis', 'Holotrichia serrata',
    'Hydrellia philippina', 'Hypolixus truncatulus', 'Leucopholis burmeisteri',
    'Libellula depressa', 'Lucilia sericata', 'Melanagromyza obtusa', 'Mylabris phalerata',
    'Oryctes rhinoceros', 'Paracoccus marginatus', 'Paradisynus rostratus', 'Parallelia algira',
    'Parasa lepida', 'Pectinophora gossypiella', 'Pelopidas mathias', 'Pempherulus affinis',
    'Pentalonia nigronervosa', 'peregrius maidis', 'Pericallia ricini', 'Perigea capensis',
    'Petrobia latens', 'Phenacoccus solenopsis', 'Phoetaliotes nebrascensis',
    'Phthorimaea operculella', 'Phyllocnistis citrella', 'Pieris brassicae', 'Pulchriphyllium',
    'Rapala varuna', 'Rastrococcus iceryoides', 'Retithrips siriacus', 'Retithrips syriacus',
    'Rhipiphorothrips cruentatus', 'Rhopalosiphum maidis', 'Rhopalosiphum padi',
    'Rhynchophorus ferrugineus', 'Riptortus pedestris', 'Sahyadrassus malabaricus',
    'Saissetia coffeae', 'Streptanus aemulans', 'sustama gremius', 'Sylepta derogata',
    'Sympetrum signiferum', 'Sympetrum vulgatum', 'Tanymecus indicus Faust',
    'Tetraneura nigriabdominalis', 'Tetrachynus cinnarinus', 'Tetranychus piercei',
    'Thalassodes quadraria', 'Thosea andamanica', 'Thrips nigripilosus', 'Thrips orientalis',
    'Thrips tabaci', 'Thysanoplusia orichalcea', 'Toxoptera odinae', 'Trialeurodes rara',
    'Trialeurodes ricini', 'Trichoplusia ni', 'Tuta absoluta', 'Udaspes folus',
    'Urentius hystricellus', 'uroleucon carthami', 'Vespula germanica', 'Xeroma mura',
    'xylosadrus compactus', 'Xylotrchus quadripes', 'Zeuzera coffe', 'non insects',
    'Papilio polytes', 'Periplaneta americana'
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mobilenetv2_insect.keras')

model = load_model()

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def ui_card(title, content):
    st.markdown(
        f"""
        <div class="card">
            <h3>{title}</h3>
            {content}
        </div>
        """, unsafe_allow_html=True
    )

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
    st.info("💡 Tip: For best accuracy, ensure the insect is well-lit and clearly visible.")

# --------------------------------------------------
# Page Definitions
# --------------------------------------------------
def intro_page():
    st.title("🐞 INSECTIFICA 🔍")
    st.subheader("AI-Powered Insect & Pest Identification")
    st.markdown("""
    **Insectifica** helps identify insects and pests instantly using artificial intelligence
    and image recognition. Designed for **students, farmers, researchers, and nature enthusiasts**.
    """)
    st.divider()
    how_it_works_section()
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("🔍 Start Identification", use_container_width=True):
            st.session_state.page = "classification"
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("ℹ️ About App"):
            st.session_state.page = "about_app"
    with col_b:
        if st.button("👨‍🔬 Developers"):
            st.session_state.page = "developers"

def about_app_page():
    st.title("ℹ️ About INSECTIFICA")
    st.markdown("""[Your about text here — same as original]""")
    st.divider()
    if st.button("➡️ Features & Use Cases"):
        st.session_state.page = "features"
    if st.button("⬅️ Back"):
        st.session_state.page = "intro"

def features_page():
    st.title("✨ Features & Use Cases")
    st.markdown("""[Your features text here — same as original]""")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👨‍🔬 Developers"):
            st.session_state.page = "developers"
    with col2:
        if st.button("⬅️ Back"):
            st.session_state.page = "about_app"

def developers_page():
    st.title("👨‍🔬 Development Team")
    st.markdown("""[Your developers text here — same as original]""")
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "intro"

def classification_page():
    st.title("🔍 Insect Classification")
    
    uploaded_file = st.file_uploader("Upload an insect image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Correct preprocessing for MobileNetV2
        img = image.resize((190, 190))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)  # Critical: use MobileNetV2 preprocessing
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing image..."):
            predictions = model.predict(img_array)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]

        if predicted_idx >= len(class_names):
            st.error("⚠️ Prediction index out of range. Please try another image.")
            return

        predicted_class = class_names[predicted_idx]
        st.success(f"**Predicted Species:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2%}")

        if predicted_class in insect_data:
            details = insect_data[predicted_class]
            st.markdown("## 🧬 Taxonomy")
            st.write(f"**Kingdom:** {details.get('Kingdom', 'N/A')}")
            st.write(f"**Phylum:** {details.get('Phylum', 'N/A')}")
            st.write(f"**Class:** {details.get('Class', 'N/A')}")
            st.write(f"**Order:** {details.get('Order', 'N/A')}")
            st.write(f"**Family:** {details.get('Family', 'N/A')}")
            st.write(f"**Genus:** {details.get('Genus', 'N/A')}")
            st.write(f"**Species:** {details.get('Species', 'N/A')}")

            st.markdown("## 🌿 Host Crops")
            st.write(details.get("Host Crops", "Not available"))

            st.markdown("## 🐛 Damage Symptoms")
            st.write(details.get("Damage Symptoms", "Not available"))

            st.markdown("## 🛡️ IPM Measures")
            st.write(details.get("IPM Measures", "Not available"))

            st.markdown("## ⚠️ Chemical Control")
            st.write(details.get("Chemical Control", "Not available"))
        else:
            st.warning("Detailed information for this species is not available in the database.")

        if st.button("⬅️ Back to Home"):
            st.session_state.page = "intro"
    else:
        st.info("Please upload an image to begin identification.")

# --------------------------------------------------
# Page Routing
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "intro"

if st.session_state.page == "intro":
    intro_page()
elif st.session_state.page == "classification":
    classification_page()
elif st.session_state.page == "about_app":
    about_app_page()
elif st.session_state.page == "features":
    features_page()
elif st.session_state.page == "developers":
    developers_page()

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown("<div class='footer'>© Department of Biotechnology | St. Joseph’s College (Autonomous), Tiruchirappalli</div>", 
            unsafe_allow_html=True)
