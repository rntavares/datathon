"""Rotas e schemas da API de predição Passos Mágicos.

Define os endpoints REST e modelos Pydantic para receber dados
de estudantes e retornar predições de risco de defasagem escolar.
"""

import logging

import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("passos_magicos")

router = APIRouter()

# ---------------------------------------------------------------------------
# Mapeamentos de encoding (espelham src/preprocessing.py)
# ---------------------------------------------------------------------------
GENERO_MAP = {"Masculino": 1, "Feminino": 0}
PEDRA_ORDINAL = {"Quartzo": 0, "Ágata": 1, "Ametista": 2, "Topázio": 3}
BINARY_SIM_NAO = {"Sim": 1, "Não": 0, "sim": 1, "não": 0}

RISK_LABELS = {
    0: "sem_defasagem",
    1: "defasagem_leve",
    2: "defasagem_severa",
}


# ---------------------------------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------------------------------
class StudentInput(BaseModel):
    """Schema de entrada para predição.

    Os campos refletem os indicadores PEDE e dados demográficos
    do dataset real. Campos opcionais permitem predição mesmo
    com dados parciais.
    """

    # Indicadores PEDE
    INDE: float | None = None
    IAA: float | None = None
    IEG: float | None = None
    IPS: float | None = None
    IDA: float | None = None
    IPP: float | None = None
    IPV: float | None = None
    IAN: float | None = None

    # Notas acadêmicas
    Matem: float | None = None
    Portug: float | None = None
    Ingles: float | None = None

    # Dados demográficos e contextuais
    Idade: int | None = None
    Genero: str | None = None
    Fase: int | None = None
    Instituicao_ensino: str | None = None
    Ano_ingresso: int | None = None

    # Classificação e avaliação
    Pedra: str | None = None
    Indicado: str | None = None
    Atingiu_PV: str | None = None


class PredictionResponse(BaseModel):
    """Schema de resposta da predição."""

    prediction: int
    risk_label: str
    probability: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    """Schema de resposta do health check."""

    status: str
    model_loaded: bool
    version: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_input(data: dict) -> dict:
    """Aplica encoding categórico nos campos de entrada.

    Espelha a lógica de ``encode_categorical`` do preprocessador
    para que o modelo receba os mesmos tipos de dados com que foi treinado.
    """
    encoded = data.copy()

    # Gênero → binário
    if encoded.get("Genero") is not None:
        encoded["Genero"] = GENERO_MAP.get(encoded["Genero"], 0)

    # Pedra → ordinal
    if encoded.get("Pedra") is not None:
        encoded["Pedra"] = PEDRA_ORDINAL.get(encoded["Pedra"], -1)

    # Indicado → binário
    if encoded.get("Indicado") is not None:
        encoded["Indicado"] = BINARY_SIM_NAO.get(encoded["Indicado"], 0)

    # Atingiu_PV → binário
    if encoded.get("Atingiu_PV") is not None:
        encoded["Atingiu_PV"] = BINARY_SIM_NAO.get(encoded["Atingiu_PV"], 0)

    return encoded


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictionResponse)
async def predict(student: StudentInput, request: Request):
    """Recebe dados de um estudante e retorna a predição de risco de defasagem."""

    model = request.app.state.model

    if model is None:
        logger.error("Modelo não carregado — impossível realizar predição")
        return JSONResponse(
            status_code=503,
            content={"detail": "Modelo não disponível. Tente novamente mais tarde."},
        )

    # Preparar DataFrame com as features esperadas pelo modelo
    raw = student.model_dump()

    # Mapear campos do input para nomes de features do modelo
    row = {
        "Fase": raw.get("Fase") or 0,
        "Gênero": GENERO_MAP.get(raw.get("Genero", ""), 0),
        "Ano ingresso": raw.get("Ano_ingresso") or 2022,
        "Pedra": PEDRA_ORDINAL.get(raw.get("Pedra", ""), -1),
        "Nº Av": 3,  # valor padrão
        "IAA": raw.get("IAA") or 0,
        "IEG": raw.get("IEG") or 0,
        "IPS": raw.get("IPS") or 0,
        "IDA": raw.get("IDA") or 0,
        "Mat": raw.get("Matem") or 0,
        "Por": raw.get("Portug") or 0,
        "IPV": raw.get("IPV") or 0,
        "IAN": raw.get("IAN") or 0,
        "IPP": raw.get("IPP") or 0,
        "anos_no_programa": (2024 - (raw.get("Ano_ingresso") or 2022)),
        "ratio_IDA_IEG": ((raw.get("IDA") or 0) / (raw.get("IEG") or 1)),
    }

    # One-hot encoding para Instituição de ensino
    inst = raw.get("Instituicao_ensino") or "Escola Pública"
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
        inst_name = col.replace("Inst_", "")
        row[col] = 1 if inst_name.lower() in inst.lower() else 0

    # Deltas temporais (0 para predição individual sem histórico)
    for delta_col in ["delta_INDE", "delta_IAA", "delta_IEG", "delta_IPS",
                      "delta_IDA", "delta_IPP", "delta_IPV", "delta_IAN",
                      "delta_Mat", "delta_Por", "delta_Ing", "pedra_evolucao"]:
        row[delta_col] = 0

    df = pd.DataFrame([row])

    # Garantir que as colunas estejam na ordem esperada pelo modelo
    expected_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else list(row.keys())
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_features]

    # Predição
    prediction = int(model.predict(df)[0])
    risk_label = RISK_LABELS.get(prediction, f"classe_{prediction}")

    # Probabilidades
    probabilities: dict[str, float] = {}
    probability = 1.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        classes = model.classes_
        for cls, p in zip(classes, proba):
            label = RISK_LABELS.get(int(cls), f"classe_{cls}")
            probabilities[label] = round(float(p), 4)
        probability = round(float(proba[list(classes).index(prediction)]), 4)
    else:
        probabilities[risk_label] = 1.0

    logger.info(
        f"Predição realizada: classe={prediction}, label={risk_label}, "
        f"probabilidade={probability}"
    )

    return PredictionResponse(
        prediction=prediction,
        risk_label=risk_label,
        probability=probability,
        probabilities=probabilities,
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Retorna o status de saúde do serviço e se o modelo está carregado."""

    model_loaded = request.app.state.model is not None
    status = "healthy" if model_loaded else "unhealthy"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        version="1.0.0",
    )
