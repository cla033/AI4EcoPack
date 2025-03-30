
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from io import StringIO
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

            importanze = model.feature_importances_
            st.write("### 🔍 Importanza delle variabili")
            for nome, val in zip(nuovo.columns, importanze):
                st.write(f"- **{nome}**: {val:.2%}")
else:
    st.info("📢 Carica un dataset per iniziare. Deve contenere: peso_kg, volume_m3, fragile, eco_priority e packaging.")


# Funzione migliorata per auto-tuning del modello
@st.cache_data(max_entries=1)
def auto_train_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X = df.drop("packaging", axis=1)
    y = df["packaging"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parametri da testare
    candidate_models = [
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
        RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42),
    ]

    best_model = None
    best_acc = 0
    for model in candidate_models:
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc > best_acc:
            best_acc = acc
            best_model = model

    return best_model, best_acc

# Override della funzione precedente (sostituzione intelligente nel contesto)
train_model = auto_train_model


# Funzione per selezionare il miglior modello
def train_best_model(df):
    X = df.drop("packaging", axis=1)
    y = df["packaging"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=150, random_state=42)
    }

    best_model = None
    best_score = 0
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
        if score > best_score:
            best_score = score
            best_model = model

    return best_model, best_score

# Dopo la previsione, calcolo CO2 e mostra grafico
if st.session_state.get("prediction_done", False):
    st.write("### 📊 Importanza delle variabili")
    importanze = st.session_state["model"].feature_importances_
    col_names = st.session_state["data"].drop("packaging", axis=1).columns
    plt.figure(figsize=(8,4))
    plt.bar(col_names, importanze)
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Calcolo CO2 simulato
    co2 = int(100 * st.session_state["input"]["peso_kg"] + 50 * st.session_state["input"]["volume_m3"])
    if st.session_state["input"]["fragile"]:
        co2 *= 1.2
    st.success(f"Stima CO₂ del packaging selezionato: {int(co2)} g")

    # Esportazione PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Risultato AI4EcoPack", ln=1, align="C")
    for k, v in st.session_state["input"].items():
        pdf.cell(200, 10, txt=f"{k}: {v}", ln=1)
    pdf.cell(200, 10, txt=f"Packaging consigliato: {st.session_state['prediction']}", ln=1)
    pdf.cell(200, 10, txt=f"CO₂ stimata: {int(co2)} g", ln=1)
    pdf_path = "/tmp/ai4ecopack_result.pdf"
    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button("📄 Scarica PDF riepilogo", data=f, file_name="ai4ecopack_result.pdf", mime="application/pdf")
