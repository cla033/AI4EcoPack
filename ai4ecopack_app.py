
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

st.set_page_config(page_title="AI4EcoPack Batch", layout="centered")
st.title("📦 AI4EcoPack - Predizione Packaging Sostenibile")

st.markdown("""
Carica un file CSV contenente le caratteristiche dei tuoi prodotti.
L'app userà un modello AI per suggerire il packaging più sostenibile per ciascuno.
""")

# Carica il modello addestrato (o addestra uno semplice se non esiste)
MODEL_PATH = "modello_rf.pkl"

def carica_o_addestra_modello():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # Dataset di esempio minimo
    data = {
        'peso_kg': [0.5, 1.2, 3.5, 5.0, 0.8],
        'volume_m3': [0.01, 0.03, 0.07, 0.1, 0.015],
        'fragile': [1, 0, 1, 0, 1],
        'eco_priority': [1, 1, 0, 0, 1],
        'packaging': ['Cartone S', 'Cartone M', 'Plastica M', 'Plastica L', 'Cartone S']
    }
    df = pd.DataFrame(data)
    X = df.drop('packaging', axis=1)
    y = df['packaging']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

modello = carica_o_addestra_modello()

# Upload CSV per predizione batch
uploaded_file = st.file_uploader("Carica file CSV con colonne: peso_kg, volume_m3, fragile, eco_priority", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if not all(col in df.columns for col in ['peso_kg', 'volume_m3', 'fragile', 'eco_priority']):
            st.error("⚠️ Il file deve contenere le colonne: peso_kg, volume_m3, fragile, eco_priority")
        else:
            predizioni = modello.predict(df)
            df['packaging_consigliato'] = predizioni
            st.success("✅ Predizioni completate con successo!")
            st.dataframe(df)

            # Esportazione CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Scarica Risultati in CSV", data=csv, file_name="risultati_packaging.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Errore durante la lettura del file: {e}")
else:
    st.info("📄 In attesa del caricamento del file CSV.")
