
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# CONFIGURAZIONE DELLA PAGINA
st.set_page_config(page_title="AI4EcoPack", page_icon="♻️", layout="wide")

# BETA badge
st.markdown("""
<style>
.fixed-bottom-right {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background-color: #e0f2f1;
    color: #00695c;
    padding: 6px 14px;
    border-radius: 10px;
    font-weight: bold;
    font-size: 16px; background-color: #a5d6a7; border: 2px solid #2e7d32;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    z-index: 9999;
}
</style>
<div class='fixed-bottom-right'><strong>BETA</strong></div>
""", unsafe_allow_html=True)

# INTESTAZIONE
st.markdown("""
<style>
    .main {
        background-color: #f4fdf4;
    }
    h1 {
        color: #2e7d32;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("♻️ AI4EcoPack")
st.subheader("Un assistente intelligente per scegliere il packaging più sostenibile 🌱")

st.markdown("---")

# CARICAMENTO DATASET
st.sidebar.header("📁 Carica il tuo dataset")
file = st.file_uploader("File CSV (deve includere la colonna 'packaging')", type=["csv"])

@st.cache_data(max_entries=1, show_spinner=True, ttl=3600)
def train_model(df):
    X = df.drop("packaging", axis=1)
    y = df["packaging"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

if file:
    df = pd.read_csv(file, low_memory=False)

    if "packaging" not in df.columns:
        st.error("❌ Il file deve contenere una colonna chiamata 'packaging'")
    else:
        st.success("✅ Dataset caricato con successo!")
        st.write("### 📊 Anteprima del dataset")
        st.dataframe(df.head())

        st.markdown("---")

        model, acc = train_model(df)
        st.success(f"🌿 Modello AI addestrato con successo! Accuratezza: **{acc:.2%}**")

        st.markdown("---")
        st.write("### 🧪 Inserisci un nuovo prodotto per ottenere una previsione")

        with st.form("form_input"):
            col1, col2 = st.columns(2)
            with col1:
                peso = st.number_input("Peso del prodotto (kg)", min_value=0.01, value=1.0, step=0.1)
                volume = st.number_input("Volume del prodotto (m3)", min_value=0.001, value=0.02, step=0.001)
            with col2:
                fragile = st.selectbox("È fragile?", [0, 1], format_func=lambda x: "Sì" if x else "No")
                eco = st.selectbox("Priorità alla sostenibilità?", [0, 1], format_func=lambda x: "Sì" if x else "No")

            submit = st.form_submit_button("♻️ Prevedi il packaging ideale")

        if submit:
            nuovo = pd.DataFrame({
                "peso_kg": [peso],
                "volume_m3": [volume],
                "fragile": [fragile],
                "eco_priority": [eco]
            })
            pred = model.predict(nuovo)[0]
            st.success(f"📦 Il packaging consigliato per questo prodotto è: **{pred}**")
            co2 = stima_co2(volume, pred)
            st.info(f"🌍 Emissioni stimate di CO₂ per questo packaging: **{co2} g**")

# Grafico dell'impatto stimato CO2
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(["Emissione CO₂ (g)"], [co2], color="#66bb6a")
ax.set_ylabel("gCO₂")
ax.set_title("🌍 Emissione stimata di CO₂ per il packaging scelto")
st.pyplot(fig)
importanze = model.feature_importances_
st.write("### 🔍 Importanza delle variabili")
for nome, val in zip(nuovo.columns, importanze):
                st.write(f"- **{nome}**: {val:.2%}")
else:
    st.info("📢 Carica un dataset per iniziare. Deve contenere: peso_kg, volume_m3, fragile, eco_priority e packaging.")

            # Grafico dell'impatto stimato CO2
import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.bar(["Emissione CO₂ (g)"], [co2], color="#66bb6a")
            ax.set_ylabel("gCO₂")
            ax.set_title("🌍 Emissione stimata di CO₂ per il packaging scelto")
            st.pyplot(fig)
