import streamlit as st
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Machine Predictive Maintenance",
    page_icon="🛠️",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Load Model ---
MODEL_PATH = "model.joblib"  # Path to the saved model file

try:
    # Try loading the model
    rfc = joblib.load(MODEL_PATH)
    st.sidebar.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.sidebar.error(f"❌ Model file '{MODEL_PATH}' not found.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# --- Page Title ---
st.title("🛠️ Machine Predictive Maintenance Classification")

st.markdown("""
Welcome to the **Machine Predictive Maintenance App**! This tool helps predict the likelihood of machine failure based on key parameters.  
Provide the required inputs below and click **'Predict Failure'** to get recommendations.
""")

# --- Input Section with Two Columns ---
col1, col2 = st.columns(2)

with col1:
    selected_type = st.selectbox('📊 Select Operational Type', ['Low', 'Medium', 'High'])
    type_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    selected_type = type_mapping[selected_type]

    process_temperature = st.number_input('🔥 Process Temperature [K]', min_value=0.0, step=0.1)

with col2:
    air_temperature = st.number_input('🌡️ Air Temperature [K]', min_value=0.0, step=0.1)
    rotational_speed = st.number_input('🔄 Rotational Speed [rpm]', min_value=0.0, step=1.0)

with col1:
    torque = st.number_input('⚙️ Torque [Nm]', min_value=0.0, step=0.1)

with col2:
    tool_wear = st.number_input('⏳ Tool Wear [min]', min_value=0.0, step=1.0)

# --- Define Prediction Labels with Recommendations ---
failure_labels = {
    0: {
        "message": "✅ No Failure - Machine is operating normally.",
        "recommendation": "No immediate action is required. Continue regular maintenance as scheduled."
    },
    1: {
        "message": "⚠️ Potential Failure - Maintenance may be required soon.",
        "recommendation": (
            "The machine shows signs of wear. Schedule preventive maintenance to avoid unexpected downtime."
        )
    },
    2: {
        "message": "❌ Critical Failure - Immediate maintenance required!",
        "recommendation": (
            "The machine is in a critical state. Perform immediate intervention to avoid significant damage."
        )
    }
}

# --- Prediction Button ---
if st.button('🔍 Predict Failure'):
    # Convert inputs into a NumPy array for prediction
    input_data = np.array([[selected_type, air_temperature, process_temperature,
                            rotational_speed, torque, tool_wear]])

    # Debugging log for input data
    st.write("Debug: Input Data ->", input_data)

    try:
        # Perform prediction
        failure_pred = rfc.predict(input_data)[0]

        # Extract message and recommendation based on prediction
        prediction = failure_labels.get(failure_pred, {
            "message": "❓ Unknown Failure Type",
            "recommendation": "No specific recommendation available."
        })

        # Display the result with a clear statement and detailed recommendation
        st.success(prediction["message"])
        st.info(f"💡 Recommendation: {prediction['recommendation']}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")

# --- Footer ---
st.markdown("""
---
Developed by [Your Name] | Powered by Streamlit
""")
