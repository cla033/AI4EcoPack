import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Set up page configuration
st.set_page_config(page_title="AI4EcoPack", page_icon=":package:", layout="centered")

# CSS for green-themed interface
st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    header {
        text-align: center;
        background-color: #1b5e20;
        color: white;
        padding: 20px 0;
    }
    h1 {
        margin: 0;
    }
    h2 {
        margin: 5px 0 20px;
    }
    main {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        padding: 20px;
    }
    section {
        background-color: #ffffff;
        border: 1px solid #c8e6c9;
        border-radius: 8px;
        padding: 20px;
        margin: 10px;
        width: 300px;
    }
    label {
        display: block;
        margin-bottom: 5px;
        font-weight: bold;
    }
    input, select, button {
        width: 100%;
        padding: 8px;
        margin-bottom: 10px;
        border: 1px solid #c8e6c9;
        border-radius: 4px;
    }
    button {
        background-color: #2e7d32;
        color: white;
        border: none;
        cursor: pointer;
    }
    button:hover {
        background-color: #1b5e20;
    }
    footer {
        text-align: center;
        padding: 10px 0;
        background-color: #1b5e20;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Function to load and preprocess data (for training purposes)
def load_data():
    # Simulated dataset based on the presentation
    data = {
        'peso_kg': [1.2, 0.5, 2.5, 3.0, 0.8],
        'volume_m3': [0.01, 0.005, 0.03, 0.04, 0.007],
        'fragile': [0, 1, 0, 0, 1],
        'eco_priority': [1, 0, 1, 1, 0],
        'packaging': ['Cartone S', 'Plastica M', 'Cartone L', 'Cartone L', 'Plastica S']
    }
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    # Convert categorical data to numerical data
    df['packaging'] = df['packaging'].astype('category').cat.codes
    return df

def train_model(df):
    X = df.drop('packaging', axis=1)
    y = df['packaging']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Model Accuracy: {accuracy:.2f}')
    
    # Save the model
    joblib.dump(model, 'random_forest_model.pkl')
    
    return model

# Load the trained model
try:
    model = joblib.load('random_forest_model.pkl')
except FileNotFoundError:
    # If the model is not found, train a new model
    df = load_data()
    df = preprocess_data(df)
    model = train_model(df)

# Main function for Streamlit app
st.title('AI4EcoPack: Packaging Suggestion System')

# Input fields
peso_kg = st.number_input('Peso (kg)', min_value=0.0, value=0.0)
volume_m3 = st.number_input('Volume (m3)', min_value=0.0, value=0.0)
fragile = st.selectbox('Fragile', options=[0, 1])
eco_priority = st.selectbox('Eco Priority', options=[0, 1])

# Predict button
if st.button('Predict Packaging'):
    features = np.array([[peso_kg, volume_m3, fragile, eco_priority]])
    prediction = model.predict(features)
    packaging_type = prediction[0]
    
    st.write(f'Suggested Packaging Type: {packaging_type}')

# Footer
st.markdown("""
    <footer>
        <p>&copy; 2025 AI4EcoPack. Tutti i diritti riservati.</p>
    </footer>
""", unsafe_allow_html=True)