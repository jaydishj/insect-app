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

/* --------------------------------------------------
   GLOBAL APP THEME
-------------------------------------------------- */
.stApp {
    background: linear-gradient(139deg, #f6fff8, #e8f5e9);
    color: #1b5e20;
    font-family: "Segoe UI", "Roboto", sans-serif;
}

/* --------------------------------------------------
   HEADINGS
-------------------------------------------------- */
h1, h2, h3, h4, h5, h6 {
    text-align: center;
    color: #1b5e20 !important;
    font-weight: 700;
}

/* --------------------------------------------------
   BUTTON STYLING
-------------------------------------------------- */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    background: linear-gradient(135deg, #2e7d32, #66bb6a);
    color: white !important;
    font-size: 18px;
    font-weight: 600;
    padding: 0.75em;
    border: none;
    transition: all 0.2s ease-in-out;
}

/* Hover effect */
.stButton > button:hover {
    background: linear-gradient(135deg, #388e3c, #81c784);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    transform: translateY(-1px);
}

/* Active (click) effect */
.stButton > button:active {
    background: linear-gradient(135deg, #1b5e20, #43a047) !important;
    transform: scale(0.97);
}

/* Disabled / loading look */
button[disabled] {
    background: linear-gradient(135deg, #1b5e20, #66bb6a) !important;
    opacity: 0.75;
    cursor: wait;
}

/* --------------------------------------------------
   FILE UPLOADER
-------------------------------------------------- */
[data-testid="stFileUploader"] {
    border: 2px dashed #2e7d32;
    border-radius: 16px;
    padding: 1em;
    background-color: #f1f8e9;
}

/* --------------------------------------------------
   IMAGES
-------------------------------------------------- */
img {
    border-radius: 16px;
    max-width: 100%;
}

/* --------------------------------------------------
   CARD UI
-------------------------------------------------- */
.card {
    background: white;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 22px;
    color: #1b5e20;
}

/* --------------------------------------------------
   FOOTER
-------------------------------------------------- */
.footer {
    text-align: center;
    font-size: 13px;
    color: #2e7d32 !important;
    margin-top: 30px;
}

/* --------------------------------------------------
   MOBILE RESPONSIVENESS
-------------------------------------------------- */
@media (max-width: 768px) {
    h1 {
        font-size: 34px !important;
    }

    .stButton > button {
        font-size: 16px;
        padding: 0.65em;
    }
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

     st.markdown("""
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

def developers_page():
     st.title("👨‍🔬 Development Team")

     st.markdown("""
    **Department of Biotechnology**  
    St. Joseph’s College (Autonomous)  
    Tiruchirappalli – 620 002  
    Contact mail id:  
    edward_bt2@mail.sjctni.edu  
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

def classification_page():
    st.title("Insect Identification")

    # ---------------- PROFESSIONAL UPLOADER CARD AT THE TOP ----------------
    st.markdown("""
    <div class="card" style="text-align: center; padding: 30px; margin-bottom: 30px; background: linear-gradient(135deg, #e8f5e9, #f1f8e9);">
        <h2 style="color: #2e7d32; margin: 0 0 15px 0;">Upload or Take a Photo</h2>
        <p style="color: #1b5e20; font-size: 17px; margin-bottom: 25px;">
            Snap a clear photo of the insect or upload from your gallery
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- CENTERED, PROFESSIONAL FILE UPLOADER ----------------
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Choose an image or use camera",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="Best results: clear, well-lit photo of one insect"
        )

    # Add a small hint for mobile users
    if 'mobile' in st.session_state.get('user_agent', '').lower():
        st.caption("Tip: On phone? Tap above to open camera directly")

    st.markdown("---")

    # ---------------- PHOTO TIPS SECTION (Below uploader) ----------------
    st.markdown("""
    <div style="background: #f8fff8; border-radius: 16px; padding: 20px; margin: 20px 0; border: 1px solid #c8e6c9;">
        <h4 style="text-align: center; color: #2e7d32; margin-top: 0;">Best Tips for Accurate Identification</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-top: 20px;">
            <div style="text-align: center; padding: 10px;">
                <div style="font-size: 45px; margin-bottom: 10px;">Focus</div>
                <b>Get Close & Sharp</b><br>
                <small>Fill the frame with the insect</small>
            </div>
            <div style="text-align: center; padding: 10px;">
                <div style="font-size: 45px; margin-bottom: 10px;">Sun</div>
                <b>Natural Daylight</b><br>
                <small>Avoid dark shadows or flash</small>
            </div>
            <div style="text-align: center; padding: 10px;">
                <div style="font-size: 45px; margin-bottom: 10px;">Angles</div>
                <b>Side & Top View</b><br>
                <small>Show wings, legs, antennae</small>
            </div>
            <div style="text-align: center; padding: 10px;">
                <div style="font-size: 45px; margin-bottom: 10px;">Background</div>
                <b>Plain Background</b><br>
                <small>Leaf, wall, or hand is ideal</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- PROCESS UPLOADED IMAGE ----------------
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # Display uploaded image
        st.markdown("<h3 style='text-align: center; color: #2e7d32;'>Uploaded Image</h3>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)

        # Preprocess and predict
        img = image.resize((190, 190))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("AI is identifying the insect..."):
            predictions = model.predict(img_array)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]

        st.markdown("---")

        if predicted_idx >= len(class_names):
            st.error("Unable to identify. Please upload a clearer image of a single insect.")
        else:
            predicted_class = class_names[predicted_idx]

            st.success(f"**Identified as:** {predicted_class}")
            st.progress(confidence)
            st.write(f"**Confidence:** {confidence:.1%}")

            # Show details
            if predicted_class in insect_data:
                details = insect_data[predicted_class]

                st.markdown("## Taxonomic Classification")
                cols = st.columns(3)
                with cols[0]:
                    st.write(f"**Kingdom:** {details.get('Kingdom', 'N/A')}")
                    st.write(f"**Phylum:** {details.get('Phylum', 'N/A')}")
                with cols[1]:
                    st.write(f"**Class:** {details.get('Class', 'N/A')}")
                    st.write(f"**Order:** {details.get('Order', 'N/A')}")
                with cols[2]:
                    st.write(f"**Family:** {details.get('Family', 'N/A')}")
                    st.write(f"**Genus:** {details.get('Genus', 'N/A')}")

                st.write(f"**Species:** {details.get('Species', 'N/A')}")

                st.markdown("## Host Crops")
                st.info(details.get("Host Crops", "Not available"))

                st.markdown("## Damage Symptoms")
                st.warning(details.get("Damage Symptoms", "Not available"))

                st.markdown("## IPM Measures")
                st.success(details.get("IPM Measures", "Not available"))

                st.markdown("## Chemical Control")
                st.error(details.get("Chemical Control", "Not available"))
            else:
                st.warning("Detailed information for this species is not yet in our database.")

        # Back button after result
        st.markdown("---")
        col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
        with col_b2:
            if st.button("Back to Home", use_container_width=True):
                st.session_state.page = "intro"
                st.rerun()

    else:
        # No image yet — show prompt
        st.info("Please upload or take a photo above to identify the insect.")

    # Always show back button at bottom
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns([1, 2, 1])
    with col_f2:
        if st.button("Back to Home", use_container_width=True, key="final_back"):
            st.session_state.page = "intro"
            st.rerun()

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
