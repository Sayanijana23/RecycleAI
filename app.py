import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RecycleAI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #e8f5e9,
        #f1fff4,
        #e3f2fd
    );
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main headings */
h1 {
    color: #176b45 !important;
    font-weight: 800 !important;
}

h2 {
    color: #176b45 !important;
}

h3 {
    color: #176b45 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #e8f5e9,
        #f1f8e9
    );
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #176b45 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.08);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        135deg,
        #11998e,
        #38ef7d
    );
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 25px;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #0f8a7f,
        #2edb6e
    );
    color: white;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.08);
    border-left: 5px solid #38a169;
}

[data-testid="stMetricLabel"] {
    color: #555 !important;
}

[data-testid="stMetricValue"] {
    color: #176b45 !important;
}

/* Alert boxes */
[data-testid="stAlert"] {
    border-radius: 15px;
}

/* Images */
[data-testid="stImage"] {
    border-radius: 15px;
}

/* Footer */
.footer-text {
    text-align: center;
    color: #4f6f5a;
    margin-top: 40px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.keras")
    return model


model = load_model()

# =========================================================
# IMAGE SETTINGS
# =========================================================

IMG_HEIGHT = 224
IMG_WIDTH = 224

class_indices = {
    "cardboard": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "plastic": 4,
    "trash": 5
}

idx2label = {
    value: key
    for key, value in class_indices.items()
}

# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def load_and_preprocess_image(image_data):

    img = Image.open(
        io.BytesIO(image_data)
    ).convert("RGB")

    img_resized = img.resize(
        (IMG_WIDTH, IMG_HEIGHT)
    )

    img_array = image.img_to_array(
        img_resized
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    return img_array, img


# =========================================================
# RECYCLING SUGGESTIONS
# =========================================================

recycling_suggestions = {

    "cardboard":
        "Flatten cardboard boxes and keep them dry. "
        "Remove food residue and place them in the "
        "appropriate cardboard recycling collection.",

    "glass":
        "Clean the glass item before recycling. "
        "Follow local recycling guidelines and avoid "
        "mixing glass with ceramics or mirrors.",

    "metal":
        "Rinse metal containers and remove leftover food. "
        "Place aluminum or steel items in the appropriate "
        "metal recycling collection.",

    "paper":
        "Keep paper clean and dry before recycling. "
        "Avoid recycling paper heavily contaminated with "
        "food, oil, or grease.",

    "plastic":
        "Rinse the plastic item to remove food residue. "
        "Check the recycling symbol and place it in the "
        "appropriate plastic recycling collection.",

    "trash":
        "This item is classified as general trash. "
        "Consider whether it can be reused or repurposed "
        "before disposing of it."
}

# =========================================================
# WASTE BIN RECOMMENDATIONS
# =========================================================

bin_recommendations = {

    "cardboard":
        "📦 Paper / Cardboard Recycling Bin",

    "glass":
        "🍾 Glass Recycling Bin",

    "metal":
        "🥫 Metal Recycling Bin",

    "paper":
        "📄 Paper Recycling Bin",

    "plastic":
        "♻️ Plastic / Recyclable Waste Bin",

    "trash":
        "🗑️ General Waste Bin"
}

# =========================================================
# HEADER
# =========================================================

st.title("♻️ RecycleAI - Intelligent Waste Classification & Recycling Assistant")

st.markdown(
    "### AI-Powered Waste Classification & Recycling Assistant"
)

st.write(
    "Upload an image of a waste item and the deep learning "
    "model will identify its category, recommend a suitable "
    "waste bin, and provide recycling advice."
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🧠 Deep Learning Model")

    st.success("MobileNetV2")

    st.divider()

    st.header("📂 Supported Classes")

    st.write("📦 **Cardboard**")
    st.write("🍾 **Glass**")
    st.write("🥫 **Metal**")
    st.write("📄 **Paper**")
    st.write("🥤 **Plastic**")
    st.write("🗑️ **Trash**")

    st.divider()

    st.info(
        "💡 Tip: Use a clear image with the waste item "
        "visible in the center."
    )

# =========================================================
# UPLOAD SECTION
# =========================================================

st.header("📸 Upload Your Waste Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG, JPEG, or PNG image."
)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image_data = uploaded_file.read()

    img_array, original_img = load_and_preprocess_image(
        image_data
    )

    # -----------------------------------------------------
    # IMAGE AND RESULT COLUMNS
    # -----------------------------------------------------

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )

    # -----------------------------------------------------
    # IMAGE PREVIEW
    # -----------------------------------------------------

    with col1:

        st.subheader("🖼️ Uploaded Image")

        st.image(
            original_img,
            use_container_width=True
        )

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    with col2:

        st.subheader("🔍 Analysis Result")

        with st.spinner(
            "🤖 AI is analyzing the image..."
        ):

            predictions = model.predict(
                img_array,
                verbose=0
            )

        predicted_class = int(
            np.argmax(
                predictions,
                axis=1
            )[0]
        )

        predicted_label = idx2label[
            predicted_class
        ]

        confidence = predictions[
            0
        ][predicted_class]

        confidence_percent = confidence * 100

        # Prediction
        st.metric(
            label="🗑️ Waste Category",
            value=predicted_label.capitalize()
        )

        st.write("")

        # Confidence
        st.metric(
            label="🎯 Prediction Confidence",
            value=f"{confidence_percent:.2f}%"
        )

        st.write("")

        # Confidence progress
        st.write("Confidence Level")

        st.progress(
            float(confidence)
        )

    # =====================================================
    # WASTE BIN RECOMMENDATION
    # =====================================================

    st.divider()

    st.subheader("🗑️ Recommended Waste Bin")

    recommended_bin = bin_recommendations.get(
        predicted_label,
        "🗑️ General Waste Bin"
    )

    st.success(
        recommended_bin
    )

    # =====================================================
    # RECYCLING RECOMMENDATION
    # =====================================================

    st.subheader("♻️ Recycling Recommendation")

    suggestion = recycling_suggestions.get(
        predicted_label,
        "No recycling suggestion available."
    )

    st.warning(
        suggestion
    )

    # =====================================================
    # PREDICTION SUMMARY
    # =====================================================

    st.divider()

    st.subheader("📊 Prediction Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.metric(
            "Category",
            predicted_label.capitalize()
        )

    with summary_col2:

        st.metric(
            "Confidence",
            f"{confidence_percent:.2f}%"
        )

    with summary_col3:

        st.metric(
            "Supported Classes",
            "6"
        )

# =========================================================
# INITIAL SCREEN
# =========================================================

else:

    st.info(
        "📷 Upload an image above to start waste classification."
    )

    st.markdown("### 🌱 How It Works")

    step1, step2, step3 = st.columns(3)

    with step1:

        st.markdown("### 1️⃣ Upload")

        st.write(
            "Upload a clear image of a waste item."
        )

    with step2:

        st.markdown("### 2️⃣ Analyze")

        st.write(
            "MobileNetV2 analyzes the uploaded image."
        )

    with step3:

        st.markdown("### 3️⃣ Classify")

        st.write(
            "Get the waste category, bin recommendation, "
            "and recycling advice."
        )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer-text">
        ♻️ <b>Smart Waste Classification</b><br>
        Made by Sayani Jana<br>
        🌱 Making waste segregation smarter
    </div>
    """,
    unsafe_allow_html=True
)