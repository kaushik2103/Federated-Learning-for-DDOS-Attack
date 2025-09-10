import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_PATH = r'reports_dl\global_model_round_19.keras'
SCALER_PATH = 'scaler.gz'
NUM_FEATURES = 50


def check_for_files():
    if not os.path.exists(SCALER_PATH):
        st.error(f"Fatal Error: Scaler file '{SCALER_PATH}' not found.")
        st.info("Please run the `prepare_dataset_dl.py` script first to generate the necessary scaler file.")
        return False
    if not os.path.exists(MODEL_PATH):
        st.error(f"Fatal Error: Model file not found at the specified path.")
        st.code(MODEL_PATH)
        st.info(
            "Please run the federated training (`server_dl.py` and `client_dl.py`) to generate a model, then update the MODEL_PATH in this script if necessary.")
        return False
    return True


st.set_page_config(layout="wide", page_title="Federated DDoS Detection System")

st.title("Federated LSTM-Based DDoS Attack Detection")
st.markdown("""
This application utilizes a federated LSTM model to classify network traffic. 
This streamlined interface provides real-time security analysis.
""")

if not check_for_files():
    st.stop()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    st.sidebar.success("Federated model and scaler loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Fatal Error: Could not load the model or scaler.")
    st.sidebar.code(f"Error details: {e}")
    st.stop()

st.sidebar.header("Network Feature Input")
st.sidebar.markdown("Enter the key traffic features below.")

feature_names_subset = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Bwd Packets',
    'Fwd Packet Length Max',
    'Bwd Packet Length Max',
    'Flow IAT Mean',
    'Average Packet Size'
]
feature_indices = {
    'Flow Duration': 0,
    'Total Fwd Packets': 1,
    'Total Bwd Packets': 2,
    'Fwd Packet Length Max': 4,
    'Bwd Packet Length Max': 8,
    'Flow IAT Mean': 11,
    'Average Packet Size': 38
}

with st.sidebar.form(key='input_form'):
    inputs = {}
    for feature in feature_names_subset:
        inputs[feature] = st.number_input(f"Enter {feature}", value=0.0, step=1.0, format="%.4f")

    predict_button = st.form_submit_button(label='Classify Traffic', use_container_width=True)

if predict_button:
    input_vector = np.zeros(NUM_FEATURES)

    for feature, value in inputs.items():
        index = feature_indices.get(feature)
        if index is not None:
            input_vector[index] = value

    scaled_vector = scaler.transform(input_vector.reshape(1, -1))

    reshaped_vector = scaled_vector.reshape(1, 1, NUM_FEATURES)

    reshaped_vector = reshaped_vector.astype(np.float32)

    with st.spinner('Analyzing traffic...'):
        prediction_tensor = model(reshaped_vector, training=False)
        prediction_proba = prediction_tensor.numpy()[0]

    predicted_class_index = np.argmax(prediction_proba)

    class_names = [
        "Benign Traffic (Class 0)",
        "Potential Malicious Traffic (Class 1)",
        "Potential Malicious Traffic (Class 2)",
        "Potential Malicious Traffic (Class 3)",
        "Potential Malicious Traffic (Class 4)"
    ]
    predicted_class_name = class_names[predicted_class_index]

    st.subheader("Classification Result")

    if predicted_class_index == 0:
        st.success(f"**Prediction:** {predicted_class_name}")
    else:
        st.error(f"**Prediction:** {predicted_class_name}")

    with st.expander("View Detailed Probabilities"):
        st.write("The model calculated the following probabilities for each class:")

        proba_df = pd.DataFrame({
            "Traffic Type": class_names,
            "Probability": prediction_proba
        })
        st.dataframe(proba_df.style.format({"Probability": "{:.2%}"}))

        st.bar_chart(proba_df.set_index("Traffic Type"))

