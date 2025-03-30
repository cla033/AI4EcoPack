
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import datetime

st.set_page_config(page_title="AI4EcoPack", layout="wide")
st.title("AI4EcoPack - Ottimizzazione del Packaging Sostenibile")

# Funzione per il caricamento dei dati
def carica_dati():
    file = st.file_uploader("Carica un file CSV con i dati dei prodotti", type=["csv"])
    if file:
        df = pd.read_csv(file)
        return df
    return None

# Funzione per addestrare il modello su grandi dataset
def addestra_modello(df):
    X = df.drop('packaging', axis=1)
    y = df['packaging']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modello = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
    modello.fit(X_train, y_train)
    y_pred = modello.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    joblib.dump(modello, "modello_packaging.pkl")
    return modello, acc, report, conf_matrix

# Caricamento del dataset
st.sidebar.header("Fase 1: Caricamento Dati")
df = carica_dati()

if df is not None:
    st.write("Anteprima del dataset:")
    st.dataframe(df.head())

    if 'packaging' not in df.columns:
        st.error("Il file deve contenere una colonna chiamata 'packaging' come etichetta target.")
    else:
        # Addestramento del modello
        st.sidebar.header("Fase 2: Addestramento del Modello")
        if st.sidebar.button("Addestra il modello"):
            with st.spinner("Addestramento in corso..."):
                modello, accuracy, report, conf_matrix = addestra_modello(df)
            st.success(f"Modello addestrato con successo. Accuracy: {accuracy:.2f}")

            # Report di classificazione
            st.subheader("Report di Classificazione")
            st.json(report)

            # Matrice di confusione
            st.subheader("Matrice di Confusione")
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

# Interfaccia per la predizione
st.sidebar.header("Fase 3: Predizione")
if os.path.exists("modello_packaging.pkl"):
    modello = joblib.load("modello_packaging.pkl")
    st.subheader("Inserisci le caratteristiche del prodotto da spedire")

    peso_kg = st.number_input("Peso (kg)", min_value=0.0, format="%f")
    altezza = st.number_input("Altezza (cm)", min_value=0.0, format="%f")
    larghezza = st.number_input("Larghezza (cm)", min_value=0.0, format="%f")
    profondita = st.number_input("Profondità (cm)", min_value=0.0, format="%f")
    fragile = st.selectbox("Fragile?", ["No", "Sì"])
    eco_priority = st.selectbox("Priorità all'eco-sostenibilità?", ["No", "Sì"])

    volume_m3 = (altezza / 100) * (larghezza / 100) * (profondita / 100)
    fragile_bin = 1 if fragile == "Sì" else 0
    eco_bin = 1 if eco_priority == "Sì" else 0

    if st.button("Consiglia Packaging"):
        nuovo = pd.DataFrame({
            'peso_kg': [peso_kg],
            'volume_m3': [volume_m3],
            'fragile': [fragile_bin],
            'eco_priority': [eco_bin]
        })
        predizione = modello.predict(nuovo)[0]
        st.success(f"Packaging consigliato: {predizione}")

        # ESG Report (semplificato)
        st.subheader("Report ESG Generato")
        data_oggi = datetime.date.today()
        report_esg = f"""
        **Data:** {data_oggi}

        **Prodotto fragile:** {'Sì' if fragile_bin else 'No'}
        **Priorità sostenibilità:** {'Sì' if eco_bin else 'No'}

        **Packaging consigliato:** {predizione}

        **CO₂ stimata risparmiata:** ~5g per spedizione
        **Materiale non riciclabile evitato:** ~10g per spedizione

        Questo suggerimento supporta gli obiettivi ESG dell'azienda riducendo sprechi e impatto ambientale.
        """
        st.markdown(report_esg)

        # Esportazione risultato
        st.download_button("Scarica risultato in CSV", data=nuovo.assign(packaging=predizione).to_csv(index=False), file_name="suggerimento_packaging.csv", mime="text/csv")
else:
    st.warning("Carica un dataset e addestra il modello per procedere con le predizioni.")
