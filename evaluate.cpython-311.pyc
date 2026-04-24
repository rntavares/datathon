"""Módulo de utilitários compartilhados do pipeline Passos Mágicos.

Fornece funções auxiliares para logging, serialização de modelos
e detecção de drift estatístico nos dados.
"""

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy.stats import ks_2samp


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configura o sistema de logging do pipeline.

    Cria e retorna um logger com formato padronizado para uso em todos
    os módulos do pipeline. Suporta saída para console e opcionalmente
    para arquivo.

    Args:
        level: Nível de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Caminho para arquivo de log. Se None, usa apenas stdout.

    Returns:
        Logger configurado com o formato e handlers especificados.
    """
    logger = logging.getLogger("passos_magicos")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Evitar duplicação de handlers ao chamar múltiplas vezes
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_model(model: Any, path: str) -> None:
    """Serializa e salva o modelo em disco usando joblib.

    Cria os diretórios pai automaticamente caso não existam.
    Registra a operação via logging.

    Args:
        model: Modelo treinado a serializar (qualquer objeto compatível com joblib).
        path: Caminho do arquivo de saída (ex: 'app/model/model.joblib').
    """
    logger = logging.getLogger("passos_magicos")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)
    logger.info(f"Modelo salvo com sucesso em: {path}")


def load_model(path: str) -> Any:
    """Carrega um modelo serializado do disco usando joblib.

    Args:
        path: Caminho do arquivo do modelo.

    Returns:
        Modelo deserializado.

    Raises:
        FileNotFoundError: Se o arquivo não existir no caminho especificado.
        ValueError: Se o arquivo não contiver um modelo válido ou estiver corrompido.
    """
    logger = logging.getLogger("passos_magicos")

    if not os.path.exists(path):
        logger.error(f"Arquivo de modelo não encontrado: {path}")
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {path}")

    try:
        model = joblib.load(path)
    except Exception as e:
        logger.error(f"Erro ao carregar modelo de {path}: {e}")
        raise ValueError(f"Arquivo de modelo inválido ou corrompido: {path}") from e

    logger.info(f"Modelo carregado com sucesso de: {path}")
    return model


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: float = 0.05,
) -> dict[str, dict]:
    """Detecta drift estatístico entre dados de referência e dados atuais.

    Utiliza o teste Kolmogorov-Smirnov (KS) para comparar a distribuição
    de cada feature numérica entre os dados de referência (treinamento)
    e os dados atuais (produção).

    Args:
        reference_data: DataFrame com distribuição de referência (treinamento).
        current_data: DataFrame com dados atuais (produção).
        threshold: p-value limiar para significância estatística (padrão: 0.05).

    Returns:
        Dicionário por feature com resultado do teste:
        {'feature_name': {'drift_detected': bool, 'p_value': float, 'statistic': float}}.
    """
    logger = logging.getLogger("passos_magicos")
    results: dict[str, dict] = {}

    # Selecionar apenas colunas numéricas presentes em ambos os DataFrames
    common_columns = reference_data.columns.intersection(current_data.columns)
    numeric_columns = [
        col
        for col in common_columns
        if pd.api.types.is_numeric_dtype(reference_data[col])
        and pd.api.types.is_numeric_dtype(current_data[col])
    ]

    for col in numeric_columns:
        ref_values = reference_data[col].dropna()
        cur_values = current_data[col].dropna()

        if len(ref_values) == 0 or len(cur_values) == 0:
            logger.warning(
                f"Feature '{col}' ignorada na detecção de drift: dados insuficientes."
            )
            continue

        statistic, p_value = ks_2samp(ref_values, cur_values)
        drift_detected = p_value < threshold

        results[col] = {
            "drift_detected": bool(drift_detected),
            "p_value": float(p_value),
            "statistic": float(statistic),
        }

        if drift_detected:
            logger.warning(
                f"Drift detectado na feature '{col}': p_value={p_value:.4f}, "
                f"statistic={statistic:.4f}"
            )

    return results
