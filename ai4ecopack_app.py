import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import tempfile
from fpdf import FPDF
import matplotlib.pyplot as plt

# Set up page configuration
st.set_page_config(page_title="AI4EcoPack", page_icon=":package:", layout="wide", initial_sidebar_state="expanded")

# CSS for improved interface and sustainability logo
st.markdown("""
    <style>
    body {
        font-family: 'Helvetica Neue', sans-serif;
        background-color: #f4f4f9;
        color: #333;
        margin: 0;
        padding: 0;
    }
    header {
        text-align: center;
        background-color: #2c3e50;
        color: white;
        padding: 20px 0;
        position: relative;
    }
    h1 {
        margin: 0;
        font-size: 3em;
    }
    h2 {
        margin: 5px 0 20px;
        font-size: 1.5em;
    }
    main {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        padding: 20px;
    }
    section {
        background-color: #ffffff;
        border: 1px solid #dcdde1;
        border-radius: 8px;
        padding: 20px;
        margin: 10px;
        width: 80%;
    }
    label {
        display: block;
        margin-bottom: 5px;
        font-weight: bold;
    }
    input, select, button {
        width: 100%;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #dcdde1;
        border-radius: 4px;
    }
    button {
        background-color: #2980b9;
        color: white;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    button:hover {
        background-color: #3498db;
    }
    footer {
        text-align: left;
        padding: 10px 20px;
        background-color: #2c3e50;
        color: white;
        font-size: 0.8em;
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
    }
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        border: 1px solid #dcdde1;
        padding: 10px;
        text-align: left;
    }
    th {
        background-color: #34495e;
        color: white;
    }
    .logo {
        position: absolute;
        top: 10px;
        right: 20px;
        width: 100px;
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
    return df

def train_model(df, use_random_search=True):
    X = df.drop('packaging', axis=1)
    y = df['packaging']
    
    # Check for null values
    if X.isnull().sum().any() or y.isnull().sum().any():
        st.error("Data contains null values. Please clean the data and try again.")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    
    if use_random_search:
        param_dist = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'min_child_weight': [1, 2, 4]
        }
        
        random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, verbose=2, n_jobs=-1)
        try:
            random_search.fit(X_train, y_train)
            model = random_search.best_estimator_
        except ValueError as e:
            st.error(f"Error during model training: {e}")
            return None
    else:
        try:
            model.fit(X_train, y_train)
        except ValueError as e:
            st.error(f"Error during model training: {e}")
            return None
    
    # Save the optimized model
    joblib.dump(model, 'optimized_xgb_model.pkl')
    
    # Evaluate the optimized model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f'Optimized Model Accuracy: {accuracy:.2f}')
    print(f'Classification Report:\n{report}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    
    return model

def update_model(dataset, model):
    features = dataset[['peso_kg', 'volume_m3', 'fragile', 'eco_priority']].values
    labels = dataset['packaging'].values
    for feature, label in zip(features, labels):
        model.fit([feature], [label])
    return model

# Load the trained model
try:
    model = joblib.load('optimized_xgb_model.pkl')
except FileNotFoundError:
    # If the model is not found, train a new model
    df = load_data()
    df = preprocess_data(df)
    model = train_model(df, use_random_search=False)

def generate_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.cell(200, 10, txt="AI4EcoPack: Packaging Suggestions", ln=True, align='C')
    
    # Add table headers
    pdf.cell(40, 10, txt="Peso (kg)", border=1)
    pdf.cell(40, 10, txt="Volume (m3)", border=1)
    pdf.cell(30, 10, txt="Fragile", border=1)
    pdf.cell(40, 10, txt="Eco Priority", border=1)
    pdf.cell(40, 10, txt="Suggested Packaging", border=1)
    pdf.cell(40, 10, txt="Environmental Impact (kg CO2)", border=1)
    pdf.ln()
    
    # Add table rows
    for _, row in df.iterrows():
        pdf.cell(40, 10, txt=str(row['peso_kg']), border=1)
        pdf.cell(40, 10, txt=str(row['volume_m3']), border=1)
        pdf.cell(30, 10, txt=str(row['fragile']), border=1)
        pdf.cell(40, 10, txt=str(row['eco_priority']), border=1)
        pdf.cell(40, 10, txt=str(row['Suggested Packaging']), border=1)
        pdf.cell(40, 10, txt=str(row['Environmental Impact (kg CO2)']), border=1)
        pdf.ln()
    
    return pdf

def generate_graph(df):
    packaging_counts = df['Suggested Packaging'].value_counts()
    fig, ax = plt.subplots()
    packaging_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Packaging Type')
    ax.set_ylabel('Count')
    ax.set_title('Total Packaging Used')
    st.pyplot(fig)
    
    # Save the figure to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        fig.savefig(tmp_file.name)
        return tmp_file.name

# Main function for Streamlit app
st.markdown("""
    <header>
        <h1>AI4EcoPack: Packaging Suggestion System</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Recycle002.svg/768px-Recycle002.svg.png" class="logo">
    </header>
""", unsafe_allow_html=True)

# Upload dataset
uploaded_file = st.file_uploader("Carica il tuo dataset CSV", type="csv")

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.write("Dataset Caricato:")
    st.write(dataset)

    if st.button('Predict Packaging for Dataset'):
        if len(dataset) < 5:
            st.error("Dataset troppo piccolo. Si prega di fornire un dataset più grande.")
        else:
            features = dataset[['peso_kg', 'volume_m3', 'fragile', 'eco_priority']].values
            try:
                predictions = model.predict(features)
                dataset['Suggested Packaging'] = predictions
                
                # Estimation of environmental impact (for simplicity, assume a fixed impact per packaging type)
                impact_factors = {
                    'Cartone S': 0.27,  # kg CO2 per unit
                    'Plastica M': 6.00, # kg CO2 per kg
                    'Cartone L': 0.27,  # kg CO2 per unit
                    'Plastica S': 6.00  # kg CO2 per kg
                }
                dataset['Environmental Impact (kg CO2)'] = dataset['Suggested Packaging'].map(impact_factors)
                
                st.write("Predictions with Environmental Impact:")
                st.dataframe(dataset)
                
                # Generate and display the graph
                graph_filename = generate_graph(dataset)
                
                # Allow download of the graph
                with open(graph_filename, "rb") as file:
                    btn = st.download_button(
                        label="Download Graph",
                        data=file,
                        file_name="packaging_graph.png",
                        mime="image/png"
                    )
                
                # Update the model with the new data
                model = update_model(dataset, model)
                joblib.dump(model, 'optimized_xgb_model.pkl')
                
                # Allow download of results as CSV
                st.download_button(label="Download Results as CSV", 
                                data=dataset.to_csv(index=False).encode('utf-8'), 
                                file_name='packaging_predictions.csv', 
                                mime='text/csv')
                
                # Generate and allow download of results as PDF
                pdf = generate_pdf(dataset)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    pdf.output(tmp_file.name)
                    st.download_button(label="Download Results as PDF", 
                                    data=tmp_file.read(), 
                                    file_name='packaging_predictions.pdf', 
                                    mime='application/pdf')
            except AttributeError as e:
                st.error(f"Error during prediction: {e}")

# Footer
st.markdown("""
    <footer>
        <p>&copy; 2025 AI4EcoPack. Tutti i diritti riservati.</p>
    </footer>
""", unsafe_allow_html=True)