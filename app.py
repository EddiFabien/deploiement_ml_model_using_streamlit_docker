import streamlit as st
import joblib
import pandas as pd

# --- Page configuration ---
st.set_page_config(page_title="Water Potability Prediction",page_icon="💧", layout="wide")

# --- Loading of model ---
@st.cache_resource
def loading_model():
    return joblib.load("models/potability_model.pkl")

# Loading
model = loading_model()

# --- Sidebar for Parameters ---
st.sidebar.title("Parameter values")

# Text inputs for water quality parameters
ph = st.sidebar.text_input("pH", "8.32")
hardness = st.sidebar.text_input("Hardness", "214.37")
solids = st.sidebar.text_input("Solids", "22018.42")
chloramines = st.sidebar.text_input("Chloramines", "8.06")
sulfate = st.sidebar.text_input("Sulfate", "356.89")
conductivity = st.sidebar.text_input("Conductivity", "363.27")
organic_carbon = st.sidebar.text_input("Organic Carbon", "18.44")
trihalomethanes = st.sidebar.text_input("Trihalomethanes", "100.34")
turbidity = st.sidebar.text_input("Turbidity", "4.63")

# Function to convert the input value and control that
def control_input(str_value):
    try:
        value = float(str_value)
    except Exception:
        value = float('NaN')
    return value

# Dictionnary of the input value
input_value = {
    'ph': control_input(ph),
    'Hardness': control_input(hardness),
    'Solids': control_input(solids),
    'Chloramines': control_input(chloramines),
    'Sulfate': control_input(sulfate),
    'Conductivity': control_input(conductivity),
    'Organic_carbon': control_input(organic_carbon),
    'Trihalomethanes': control_input(trihalomethanes),
    'Turbidity': control_input(turbidity)
}
# Create input DataFrame with feature names in the same order
input_value = pd.DataFrame([input_value])


# --- Make prediction ---
try:
    pred = model.predict(input_value)
except Exception:
    pred = "Unknown"

# --- Button to Predict ---
calcul = st.sidebar.button("Calcul")

# --- Main Content ---
st.markdown(
    "<h2 style='text-align: center; color: white;'>Prediction of the water potability</h2>",
    unsafe_allow_html=True,
)

# Image path (replace with the correct image path if needed)
image_path = "images/_.jpeg"  # Ensure image.png is in the same folder or provide full path

# Display Results when Predict is pressed
if calcul:
    # Example of a model output (you can replace this logic with an ML model)
    potability = pred[0]

    # Determine potability
    potability = "Potable" if potability == 1 else "Not Potable"

    # Safely calculate probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_value)
        probability = proba[0][1] * 100  # Probability for class 1
    else:
        probability = "Unknown"  # Default probability

    # Determine probability of belonging
    probability = "With " + str(round(probability, 2)) + " %" + " probability of belonging."

    col1, col2, col3 = st.columns([1, 2, 1])  # Center content

    with col2:
        st.image(image_path, use_container_width=True, caption="")
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 10px;">
                <h3>{potability}</h3>
                <p style="font-size: 18px;">{probability}</p>
                <em style="color: gray;">Health is the primary wealth</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

