import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import tempfile
from fpdf import FPDF
import plotly.express as px
import plotly.io as pio

# Set up page configuration
st.set_page_config(page_title="AI4EcoPack", page_icon=":package:", layout="wide")

# CSS for improved interface and sustainability logo
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    header {
        text-align: center;
        color: #1b5e20;
        padding: 20px 0;
        position: relative;
    }
    h1 {
        margin: 0;
        font-size: 3em;
        color: #1b5e20;
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
        border: 1px solid #c8e6c9;
        border-radius: 8px;
        padding: 20px;
        margin: 10px;
        width: 80%.
    }
    label {
        display: block.
        margin-bottom: 5px.
        font-weight: bold.
    }
    input, select, button, .stSelectbox {
        width: 100%.
        padding: 8px.
        margin-bottom: 10px.
        border: 1px solid #000000.
        border-radius: 4px.
    }
    button {
        background-color: #2e7d32.
        color: white.
        border: none.
        cursor: pointer.
    }
    button:hover {
        background-color: #1b5e20.
    }
    .stSelectbox div[data-baseweb="select"] {
        border: 1px solid #000000;
    }
    .green-material {
        border: 2px solid #2e7d32;
        background-color: #e8f5e9;
    }
    .red-material {
        border: 2px solid #b71c1c;
        background-color: #ffebee;
    }
    footer {
        text-align: left.
        padding: 10px 20px.
        background-color: #1b5e20.
        color: white.
        font-size: 0.8em.
        position: fixed.
        bottom: 0.
        left: 0.
        width: 100%.
    }
    table {
        width: 100%.
        border-collapse: collapse.
    }
    th, td {
        border: 1px solid #c8e6c9.
        padding: 8px.
        text-align: left.
    }
    th {
        background-color: #2e7d32.
        color: white.
    }
    .logo {
        position: absolute.
        top: 10px.
        right: 20px.
        width: 100px.
    }
    </style>
""", unsafe_allow_html=True)

# Function to load and preprocess data (for training purposes)
def load_data():
    # Simulated dataset based on the presentation
    data = {
        'peso_kg': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        'volume_m3': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05],
        'fragile': [0, 1, 0, 0, 1, 0, 1, 1, 0, 1],
        'eco_priority': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        'packaging': ['Cartone XS', 'Plastica XS', 'Cartone S', 'Plastica S', 'Cartone M', 'Plastica M', 'Cartone L', 'Plastica L', 'Cartone XL', 'Plastica XL']
    }
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    # Convert categorical data to numerical data
    df['packaging'] = df['packaging'].astype('category').cat.codes
    return df

def train_model(df, use_grid_search=True):
    X = df.drop('packaging', axis=1)
    y = df['packaging']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check if the dataset is large enough for Grid Search
    if use_grid_search and len(X_train) < 5:
        st.warning("Dataset too small for Grid Search. Training with default parameters.")
        use_grid_search = False
    
    if use_grid_search:
        # Hyperparameter tuning using Grid Search
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        
        # Best parameters
        best_params = grid_search.best_params_
        print(f'Best Parameters: {best_params}')
        
        # Train the optimized model
        optimized_model = RandomForestClassifier(**best_params, random_state=42)
    else:
        # Train the model with default parameters
        optimized_model = RandomForestClassifier(random_state=42)
    
    optimized_model.fit(X_train, y_train)
    
    # Save the optimized model
    joblib.dump(optimized_model, 'optimized_random_forest_model.pkl')
    
    # Evaluate the optimized model
    y_pred = optimized_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f'Optimized Model Accuracy: {accuracy:.2f}')
    print(f'Classification Report:\n{report}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    
    return optimized_model

# Load the trained model
try:
    model = joblib.load('optimized_random_forest_model.pkl')
except FileNotFoundError:
    # If the model is not found, train a new model
    df = load_data()
    df = preprocess_data(df)
    model = train_model(df, use_grid_search=False)

# Function to select the packaging size
def select_packaging_size(volume_m3):
    if volume_m3 <= 0.01:
        return 'XS'
    elif volume_m3 <= 0.05:
        return 'S'
    elif volume_m3 <= 0.1:
        return 'M'
    elif volume_m3 <= 0.5:
        return 'L'
    else:
        return 'XL'

# Function to select packaging material based on user input and environmental impact
def select_packaging_material(material):
    impact_factors = {
        'Plastic': {'XS': 5.00, 'S': 6.00, 'M': 7.00, 'L': 8.00, 'XL': 9.00},
        'Glass': {'XS': 1.50, 'S': 1.75, 'M': 2.00, 'L': 2.50, 'XL': 3.00},
        'Cardboard': {'XS': 0.20, 'S': 0.27, 'M': 0.35, 'L': 0.45, 'XL': 0.55},
        'Bamboo': {'XS': 0.10, 'S': 0.12, 'M': 0.15, 'L': 0.18, 'XL': 0.20},
        'Metal': {'XS': 1.00, 'S': 1.20, 'M': 1.50, 'L': 1.80, 'XL': 2.00},
        'Paper': {'XS': 0.05, 'S': 0.07, 'M': 0.10, 'L': 0.12, 'XL': 0.15},
        'Foam': {'XS': 0.50, 'S': 0.60, 'M': 0.70, 'L': 0.80, 'XL': 0.90},
        'Biodegradable': {'XS': 0.30, 'S': 0.35, 'M': 0.40, 'L': 0.50, 'XL': 0.60}
    }
    return impact_factors[material]

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

def generate_chart(df):
    # Count the total packaging used divided by material and size
    packaging_counts = df['Suggested Packaging'].value_counts().reset_index()
    packaging_counts.columns = ['Packaging', 'Count']
    
    # Calculate the total pollution produced
    total_pollution = df['Environmental Impact (kg CO2)'].sum()
    
    # Create the bar chart for packaging counts
    fig = px.bar(packaging_counts, x='Packaging', y='Count', 
                 labels={'Packaging': 'Packaging Type', 'Count': 'Total Used'},
                 title='Total Packaging Used by Type and Size')
    
    # Add total pollution as a text annotation
    fig.add_annotation(
        text=f"Total Pollution: {total_pollution:.2f} kg CO2",
        xref="paper", yref="paper",
        x=0.95, y=0.95, showarrow=False,
        font=dict(size=14, color="red")
    )
    
    fig.update_layout(
        xaxis_title="Packaging Type",
        yaxis_title="Total Used",
        font=dict(size=12, family="Arial, sans-serif"),
        title_font=dict(size=16, family="Arial, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    
    return fig

# Main function for Streamlit app
st.markdown("""
    <header>
        <h1>AI4EcoPack: Packaging Suggestion System 🌍📦</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Recycle002.svg/768px-Recycle002.svg.png" class="logo">
    </header>
""", unsafe_allow_html=True)

# Allow user to select packaging materials
materials = st.multiselect(
    'Select packaging materials: 🌱',
    ['Plastic', 'Glass', 'Cardboard', 'Bamboo', 'Metal', 'Paper', 'Foam', 'Biodegradable'],
    default=['Plastic', 'Cardboard']
)

# Define packaging types based on selected materials
packaging_options = []
for material in materials:
    packaging_options.extend([f'{material} XS', f'{material} S', f'{material} M', f'{material} L', f'{material} XL'])

# Mapping for packaging types
packaging_mapping = {i: option for i, option in enumerate(packaging_options)}

# Upload dataset
uploaded_file = st.file_uploader("Carica il tuo dataset CSV 📂", type="csv")

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.write("Dataset Caricato 📈:")
    st.write(dataset)

    if st.button('Predict Packaging for Dataset 🚀'):
        if len(dataset) < 5:
            st.error("Dataset troppo piccolo. Si prega di fornire un dataset più grande.")
        else:
            features = dataset[['peso_kg', 'volume_m3', 'fragile', 'eco_priority']].values
            predictions = model.predict(features)
            dataset['Suggested Packaging'] = [packaging_mapping[p] for p in predictions if p in packaging_mapping]

            # Select packaging size based on volume
            dataset['Suggested Packaging'] = dataset.apply(lambda row: f"{row['Suggested Packaging'].split()[0]} {select_packaging_size(row['volume_m3'])}", axis=1)
            
            # Select packaging material based on user input and calculate environmental impact
            dataset['Environmental Impact (kg CO2)'] = 0
            for material in materials:
                impact_factors = select_packaging_material(material)
                for size in ['XS', 'S', 'M', 'L', 'XL']:
                    mask = (dataset['Suggested Packaging'] == f'{material} {size}')
                    dataset.loc[mask, 'Environmental Impact (kg CO2)'] = impact_factors[size]
            
            # Apply green/red color based on environmental impact
            dataset['Impact Color'] = dataset['Environmental Impact (kg CO2)'].apply(lambda x: 'green-material' if x < 1 else 'red-material')
            
            st.write("Predictions with Environmental Impact 🌿:")
            st.dataframe(dataset[['peso_kg', 'volume_m3', 'fragile', 'eco_priority', 'Suggested Packaging', 'Environmental Impact (kg CO2)']])
            
            # Generate and display chart
            fig = generate_chart(dataset)
            st.plotly_chart(fig)
            
            # Allow download of chart as image
            try:
                img_bytes = pio.to_image(fig, format="png")
                st.download_button(label="Download Chart as PNG 🖼️", 
                                   data=img_bytes, 
                                   file_name='packaging_chart.png', 
                                   mime='image/png')
            except ValueError as e:
                st.error(f"Error generating image: {e}")
            
            # Allow download of results as CSV
            st.download_button(label="Download Results as CSV 📄", 
                               data=dataset.to_csv(index=False).encode('utf-8'), 
                               file_name='packaging_predictions.csv', 
                               mime='text/csv')
            
            # Generate and allow download of results as PDF
            pdf = generate_pdf(dataset)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                pdf.output(tmp_file.name)
                st.download_button(label="Download Results as PDF 📑", 
                                   data=tmp_file.read(), 
                                   file_name='packaging_predictions.pdf', 
                                   mime='application/pdf')

# Footer
st.markdown("""
    <footer>
        <p>&copy; 2025 AI4EcoPack. Tutti i diritti riservati.</p>
    </footer>
""", unsafe_allow_html=True)
