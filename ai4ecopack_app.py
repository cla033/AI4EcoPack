
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI4EcoPack", layout="centered")
st.title("📦 AI4EcoPack: Packaging sostenibile con AI")

st.markdown("Carica il tuo file CSV per addestrare il modello e prevedere il packaging ottimale.")

file = st.file_uploader("Carica un file CSV", type=["csv"])

@st.cache_data
def train_model(df):
    X = df.drop("packaging", axis=1)
    y = df["packaging"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

if file:
    df = pd.read_csv(file)
    if "packaging" not in df.columns:
        st.error("❌ Il CSV deve contenere una colonna chiamata 'packaging'")
    else:
        st.success("✅ Dataset caricato con successo!")
        st.write("Anteprima del dataset:", df.head())

        model, acc = train_model(df)
        st.success(f"Modello AI addestrato! Accuratezza: {acc:.2%}")

        st.subheader("🔍 Inserisci un nuovo prodotto da analizzare")

        peso = st.number_input("Peso (kg)", 0.1, 100.0, step=0.1)
        volume = st.number_input("Volume (m3)", 0.001, 10.0, step=0.001)
        fragile = st.selectbox("È fragile?", [0, 1], format_func=lambda x: "Sì" if x else "No")
        eco = st.selectbox("Priorità alla sostenibilità?", [0, 1], format_func=lambda x: "Sì" if x else "No")

        if st.button("Prevedi packaging ottimale"):
            nuovo = pd.DataFrame({
                "peso_kg": [peso],
                "volume_m3": [volume],
                "fragile": [fragile],
                "eco_priority": [eco]
            })
            pred = model.predict(nuovo)[0]
            st.success(f"📦 Packaging consigliato: **{pred}**")
