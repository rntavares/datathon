"""Aplicação Streamlit — Passos Mágicos ML Dashboard.

Interface visual para predição de risco de defasagem escolar,
visualização de métricas do modelo e monitoramento de drift.

Executar: streamlit run streamlit_app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Garantir imports do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import load_model, detect_drift

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Passos Mágicos ML",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MODEL_PATH = "app/model/model.joblib"
METADATA_PATH = "app/model/metadata.json"

PEDRA_MAP = {"Quartzo": 0, "Ágata": 1, "Ametista": 2, "Topázio": 3}
GENERO_MAP = {"Masculino": 1, "Feminino": 0}
RISK_LABELS = {0: "✅ Sem Defasagem", 1: "⚠️ Em Risco de Defasagem"}
RISK_COLORS = {0: "#27ae60", 1: "#e74c3c"}


# ---------------------------------------------------------------------------
# Cache de modelo e metadados
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    try:
        return load_model(MODEL_PATH)
    except Exception:
        return None


@st.cache_data
def get_metadata():
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Sidebar — Navegação
# ---------------------------------------------------------------------------
st.sidebar.image("https://passosmagicos.org.br/wp-content/uploads/2024/06/logo-passos-magicos.png", width=200)
st.sidebar.title("🔮 Passos Mágicos ML")
page = st.sidebar.radio(
    "Navegação",
    ["🎯 Predição de Risco", "📊 Métricas do Modelo", "🔍 Monitoramento de Drift"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Datathon Pós Tech — FIAP")
st.sidebar.caption("Case Passos Mágicos 2024")


# ---------------------------------------------------------------------------
# Página 1: Predição de Risco
# ---------------------------------------------------------------------------
if page == "🎯 Predição de Risco":
    st.title("🎯 Predição de Risco de Defasagem Escolar")
    st.markdown(
        "Preencha os dados do estudante para estimar o risco de defasagem escolar "
        "com base nos indicadores PEDE."
    )

    model = get_model()
    if model is None:
        st.error("⚠️ Modelo não encontrado. Execute o treinamento primeiro.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📋 Dados do Estudante")
        idade = st.number_input("Idade", min_value=6, max_value=25, value=12)
        genero = st.selectbox("Gênero", ["Masculino", "Feminino"])
        fase = st.number_input("Fase (nível no programa)", min_value=0, max_value=7, value=3)
        ano_ingresso = st.number_input("Ano de Ingresso", min_value=2010, max_value=2024, value=2021)
        pedra = st.selectbox("Classificação Pedra", ["Quartzo", "Ágata", "Ametista", "Topázio"], index=2)

    with col2:
        st.subheader("📈 Indicadores PEDE")
        iaa = st.slider("IAA (Auto Avaliação)", 0.0, 10.0, 7.5, 0.1)
        ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 7.0, 0.1)
        ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 6.5, 0.1)
        ida = st.slider("IDA (Aprendizagem)", 0.0, 10.0, 7.0, 0.1)
        ipp = st.slider("IPP (Psicopedagógico)", 0.0, 10.0, 6.0, 0.1)
        ipv = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, 7.0, 0.1)
        ian = st.slider("IAN (Adequação ao Nível)", 0.0, 10.0, 6.5, 0.1)

    with col3:
        st.subheader("📝 Notas Acadêmicas")
        mat = st.slider("Matemática", 0.0, 10.0, 7.0, 0.1)
        por = st.slider("Português", 0.0, 10.0, 7.5, 0.1)

    st.markdown("---")

    if st.button("🔮 Realizar Predição", type="primary", use_container_width=True):
        # Montar features na ordem esperada pelo modelo
        row = {
            "Fase": fase,
            "Gênero": GENERO_MAP[genero],
            "Ano ingresso": ano_ingresso,
            "Pedra": PEDRA_MAP[pedra],
            "Nº Av": 3,
            "IAA": iaa, "IEG": ieg, "IPS": ips, "IDA": ida,
            "Mat": mat, "Por": por,
            "IPV": ipv, "IAN": ian, "IPP": ipp,
            "anos_no_programa": 2024 - ano_ingresso,
            "ratio_IDA_IEG": ida / ieg if ieg > 0 else 0,
        }

        # One-hot para instituição (default: Escola Pública)
        inst_cols = [
            "Inst_Concluiu o 3º EM", "Inst_Escola JP II", "Inst_Escola Pública",
            "Inst_Nenhuma das opções acima", "Inst_Privada",
            "Inst_Privada *Parcerias com Bolsa 100%",
            "Inst_Privada - Pagamento por *Empresa Parceira",
            "Inst_Privada - Programa de Apadrinhamento",
            "Inst_Privada - Programa de apadrinhamento",
            "Inst_Pública", "Inst_Rede Decisão",
        ]
        for col in inst_cols:
            row[col] = 1 if col == "Inst_Escola Pública" else 0

        # Deltas temporais (0 para predição individual)
        for delta in ["delta_INDE", "delta_IAA", "delta_IEG", "delta_IPS",
                      "delta_IDA", "delta_IPP", "delta_IPV", "delta_IAN",
                      "delta_Mat", "delta_Por", "delta_Ing", "pedra_evolucao"]:
            row[delta] = 0

        df = pd.DataFrame([row])
        expected = model.feature_names_in_ if hasattr(model, "feature_names_in_") else list(row.keys())
        for col in expected:
            if col not in df.columns:
                df[col] = 0
        df = df[expected]

        prediction = int(model.predict(df)[0])
        proba = model.predict_proba(df)[0] if hasattr(model, "predict_proba") else [1.0]

        # Exibir resultado
        risk_label = RISK_LABELS.get(prediction, f"Classe {prediction}")
        risk_color = RISK_COLORS.get(prediction, "#333")

        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            st.markdown(
                f"""
                <div style="text-align:center; padding:30px; border-radius:15px;
                    background:linear-gradient(135deg, {risk_color}22, {risk_color}11);
                    border:2px solid {risk_color};">
                    <h1 style="color:{risk_color}; margin:0;">{risk_label}</h1>
                    <p style="font-size:1.3rem; color:#555; margin-top:10px;">
                        Probabilidade: <strong>{max(proba)*100:.1f}%</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Probabilidade Sem Risco", f"{proba[0]*100:.1f}%")
        with c2:
            if len(proba) > 1:
                st.metric("Probabilidade Em Risco", f"{proba[1]*100:.1f}%")


# ---------------------------------------------------------------------------
# Página 2: Métricas do Modelo
# ---------------------------------------------------------------------------
elif page == "📊 Métricas do Modelo":
    st.title("📊 Métricas do Modelo Treinado")

    metadata = get_metadata()
    if not metadata:
        st.warning("Metadados do modelo não encontrados.")
        st.stop()

    metrics = metadata.get("metrics", {})
    cv = metadata.get("cross_validation", {})

    # Cards de métricas
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
    m2.metric("Precision", f"{metrics.get('precision', 0)*100:.1f}%")
    m3.metric("Recall", f"{metrics.get('recall', 0)*100:.1f}%")
    m4.metric("F1-Score", f"{metrics.get('f1_score', 0)*100:.1f}%")

    st.markdown("---")

    # Informações do modelo
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ℹ️ Informações do Modelo")
        st.write(f"**Algoritmo:** {metadata.get('algorithm', 'N/A')}")
        st.write(f"**Versão:** {metadata.get('version', 'N/A')}")
        st.write(f"**Treinado em:** {metadata.get('trained_at', 'N/A')}")
        st.write(f"**Amostras de treino:** {metadata.get('training_samples', 'N/A')}")
        st.write(f"**Amostras de teste:** {metadata.get('test_samples', 'N/A')}")
        st.write(f"**Tipo de classificação:** {metadata.get('classification_type', 'N/A')}")

    with col2:
        st.subheader("🔄 Validação Cruzada (5 folds)")
        if cv:
            cv_data = []
            for metric_name, values in cv.items():
                cv_data.append({
                    "Métrica": metric_name.replace("_", " ").title(),
                    "Média": f"{values['mean']*100:.2f}%",
                    "Desvio Padrão": f"±{values['std']*100:.2f}%",
                })
            st.dataframe(pd.DataFrame(cv_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📋 Features Utilizadas")
    features = metadata.get("features", [])
    if features:
        cols = st.columns(3)
        for i, feat in enumerate(features):
            cols[i % 3].write(f"• {feat}")


# ---------------------------------------------------------------------------
# Página 3: Monitoramento de Drift
# ---------------------------------------------------------------------------
elif page == "🔍 Monitoramento de Drift":
    st.title("🔍 Monitoramento de Drift")
    st.markdown(
        "Detecção de drift estatístico entre dados de referência (treinamento) "
        "e dados atuais usando o teste Kolmogorov-Smirnov."
    )

    # Gerar dados de amostra para demonstração
    np.random.seed(42)
    n = 200
    features = ["INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]

    ref = pd.DataFrame({f: np.random.normal(5.0, 1.5, n) for f in features})
    cur = pd.DataFrame({f: np.random.normal(5.0, 1.5, n) for f in features})
    # Drift artificial em IDA e IPV
    cur["IDA"] = np.random.normal(6.5, 2.0, n)
    cur["IPV"] = np.random.normal(3.5, 1.0, n)

    results = detect_drift(ref, cur, threshold=0.05)

    # Tabela de resultados
    drift_data = []
    for feat, res in sorted(results.items()):
        detected = res["drift_detected"]
        p_val = res["p_value"]
        if detected:
            status = "🔴 Drift Detectado"
        elif p_val < 0.1:
            status = "🟡 Atenção"
        else:
            status = "🟢 Normal"
        drift_data.append({
            "Feature": feat,
            "p-value": f"{p_val:.6f}",
            "Estatística KS": f"{res['statistic']:.4f}",
            "Status": status,
        })

    st.dataframe(pd.DataFrame(drift_data), use_container_width=True, hide_index=True)

    # Resumo visual
    st.markdown("---")
    n_drift = sum(1 for r in results.values() if r["drift_detected"])
    n_ok = len(results) - n_drift

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de Features", len(results))
    c2.metric("🟢 Sem Drift", n_ok)
    c3.metric("🔴 Com Drift", n_drift)

    if n_drift > 0:
        st.warning(
            f"⚠️ Drift detectado em {n_drift} feature(s): "
            f"{', '.join(f for f, r in results.items() if r['drift_detected'])}. "
            "Considere retreinar o modelo."
        )
    else:
        st.success("✅ Nenhum drift significativo detectado.")
