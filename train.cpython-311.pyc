"""Módulo de avaliação do pipeline Passos Mágicos.

Responsável por calcular métricas de desempenho do modelo treinado,
gerar matriz de confusão, relatório de classificação e verificar
se o modelo atende ao limiar mínimo de qualidade.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger("passos_magicos")


def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calcula métricas de avaliação do modelo.

    Gera predições no conjunto de teste e calcula accuracy, precision,
    recall e f1_score utilizando média ponderada para cenários multi-classe.

    Args:
        model: Modelo treinado com método predict.
        X_test: Features de teste.
        y_test: Labels de teste.

    Returns:
        Dicionário com métricas: {'accuracy': float, 'precision': float,
        'recall': float, 'f1_score': float}.
    """
    logger.info(f"Avaliando modelo com {X_test.shape[0]} amostras de teste")

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    logger.info(
        f"Métricas de avaliação — accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
        f"f1_score={metrics['f1_score']:.4f}"
    )
    return metrics


def generate_confusion_matrix(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> np.ndarray:
    """Gera a matriz de confusão.

    Args:
        model: Modelo treinado com método predict.
        X_test: Features de teste.
        y_test: Labels de teste.

    Returns:
        Matriz de confusão como numpy array.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"Matriz de confusão gerada ({cm.shape[0]}x{cm.shape[1]})")
    logger.info(f"\n{cm}")
    return cm


def generate_classification_report(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> str:
    """Gera relatório de classificação completo.

    Utiliza sklearn classification_report para produzir um relatório
    formatado com precision, recall, f1-score e support por classe.

    Args:
        model: Modelo treinado com método predict.
        X_test: Features de teste.
        y_test: Labels de teste.

    Returns:
        String com relatório de classificação formatado.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)

    logger.info("Relatório de classificação gerado:")
    logger.info(f"\n{report}")
    return report


def check_quality_threshold(
    metrics: dict[str, float],
    threshold: float = 0.7,
    primary_metric: str = "f1_score",
) -> bool:
    """Verifica se o modelo atende ao limiar mínimo de qualidade.

    Compara o valor da métrica principal com o threshold configurado.
    Emite alerta via logging se o modelo não atender ao critério.

    Args:
        metrics: Dicionário de métricas calculadas (ex: saída de evaluate_model).
        threshold: Limiar mínimo aceitável (padrão: 0.7).
        primary_metric: Métrica principal para avaliação (padrão: 'f1_score').

    Returns:
        True se o modelo atende ao limiar, False caso contrário.
    """
    value = metrics.get(primary_metric)

    if value is None:
        logger.warning(
            f"Métrica '{primary_metric}' não encontrada nas métricas fornecidas. "
            f"Métricas disponíveis: {list(metrics.keys())}"
        )
        return False

    passes = value >= threshold

    if passes:
        logger.info(
            f"Modelo atende ao limiar de qualidade: "
            f"{primary_metric}={value:.4f} >= {threshold}"
        )
    else:
        logger.warning(
            f"Modelo NÃO atende ao limiar de qualidade: "
            f"{primary_metric}={value:.4f} < {threshold}"
        )

    return passes
