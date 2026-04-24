"""Módulo de treinamento do pipeline Passos Mágicos.

Responsável por treinar modelos de classificação scikit-learn,
realizar validação cruzada e orquestrar o pipeline completo de
treinamento incluindo serialização do modelo e metadados.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

from src.evaluate import evaluate_model
from src.utils import save_model

logger = logging.getLogger("passos_magicos")

# Mapeamento de algoritmos suportados para suas classes scikit-learn.
_ALGORITHM_MAP: dict[str, type] = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
}

# Hiperparâmetros padrão por algoritmo.
_DEFAULT_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 100,
        "random_state": 42,
        "class_weight": "balanced",
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "random_state": 42,
    },
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": 42,
        "class_weight": "balanced",
    },
}


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algorithm: str = "random_forest",
    hyperparams: dict | None = None,
) -> Any:
    """Treina um modelo de classificação scikit-learn.

    Suporta os algoritmos Random Forest, Gradient Boosting e Logistic Regression.
    Utiliza class_weight='balanced' quando disponível para lidar com dados
    desbalanceados.

    Args:
        X_train: Features de treinamento.
        y_train: Labels de treinamento.
        algorithm: Algoritmo a utilizar ('random_forest', 'gradient_boosting',
                   'logistic_regression').
        hyperparams: Hiperparâmetros customizados. Se None, usa padrões otimizados.

    Returns:
        Modelo treinado (instância de BaseEstimator).

    Raises:
        ValueError: Se o algoritmo não for suportado.
    """
    if algorithm not in _ALGORITHM_MAP:
        raise ValueError(
            f"Algoritmo '{algorithm}' não suportado. "
            f"Opções: {list(_ALGORITHM_MAP.keys())}"
        )

    # Montar hiperparâmetros: padrão + customizados
    params = _DEFAULT_HYPERPARAMS.get(algorithm, {}).copy()
    if hyperparams is not None:
        params.update(hyperparams)

    logger.info(f"Treinando modelo com algoritmo '{algorithm}'")
    logger.info(f"Hiperparâmetros: {params}")
    logger.info(
        f"Dados de treinamento: {X_train.shape[0]} amostras, "
        f"{X_train.shape[1]} features"
    )

    model_class = _ALGORITHM_MAP[algorithm]
    model = model_class(**params)

    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time

    logger.info(f"Treinamento concluído em {elapsed:.2f}s")
    return model


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
) -> dict[str, list[float]]:
    """Realiza validação cruzada e retorna métricas por fold.

    Utiliza sklearn cross_val_score com múltiplas métricas de scoring
    para avaliar a robustez do modelo em diferentes partições dos dados.

    Args:
        model: Modelo scikit-learn a validar.
        X: Features completas.
        y: Labels completas.
        n_folds: Número de folds (mínimo 2).

    Returns:
        Dicionário com listas de scores por fold:
        {'accuracy': [...], 'f1_weighted': [...],
         'precision_weighted': [...], 'recall_weighted': [...]}.
    """
    logger.info(f"Iniciando validação cruzada com {n_folds} folds")

    scoring_metrics = [
        "accuracy",
        "f1_weighted",
        "precision_weighted",
        "recall_weighted",
    ]

    results: dict[str, list[float]] = {}

    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, cv=n_folds, scoring=metric)
        results[metric] = scores.tolist()
        logger.info(
            f"  {metric}: média={np.mean(scores):.4f} ± {np.std(scores):.4f}"
        )

    logger.info("Validação cruzada concluída")
    return results


def run_training_pipeline(
    df: pd.DataFrame,
    target_column: str = "Defasagem",
    test_size: float = 0.2,
    model_output_path: str = "app/model/model.joblib",
) -> tuple[Any, dict]:
    """Executa o pipeline completo de treinamento.

    Etapas:
    1. Transformar variável alvo para classificação binária
       (defasagem < 0 → 1 "em_risco", >= 0 → 0 "sem_risco")
    2. Separar features e target, realizar train/test split
    3. Treinar modelo com Random Forest
    4. Realizar validação cruzada
    5. Avaliar modelo nos dados de teste
    6. Salvar modelo serializado e metadados

    Args:
        df: DataFrame com features e coluna alvo.
        target_column: Nome da coluna alvo (padrão: 'Defasagem').
        test_size: Proporção dos dados para teste (padrão: 0.2).
        model_output_path: Caminho para salvar o modelo serializado.

    Returns:
        Tupla (modelo_treinado, métricas_de_avaliação).

    Raises:
        ValueError: Se a coluna alvo não existir no DataFrame.
    """
    logger.info("=== Iniciando pipeline de treinamento ===")

    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo '{target_column}' não encontrada no DataFrame")

    df = df.copy()

    # 1. Transformar target para binário
    logger.info("Transformando variável alvo para classificação binária")
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=[target_column])

    y = (df[target_column] < 0).astype(int)  # 1 = em_risco, 0 = sem_risco
    X = df.drop(columns=[target_column])

    # Remover colunas não numéricas restantes
    X = X.select_dtypes(include=[np.number])

    # Preencher NaN restantes com 0 para evitar erros no treinamento
    X = X.fillna(0)

    logger.info(
        f"Distribuição do target: sem_risco={int((y == 0).sum())}, "
        f"em_risco={int((y == 1).sum())}"
    )

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info(
        f"Split: treino={X_train.shape[0]} amostras, "
        f"teste={X_test.shape[0]} amostras"
    )

    # 3. Treinar modelo
    model = train_model(X_train, y_train, algorithm="random_forest")

    # 4. Validação cruzada
    cv_results = cross_validate_model(model, X_train, y_train, n_folds=5)

    # 5. Avaliar no conjunto de teste
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"Métricas no conjunto de teste: {metrics}")

    # 6. Salvar modelo e metadados
    save_model(model, model_output_path)

    metadata = {
        "algorithm": "random_forest",
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target_column": target_column,
        "classification_type": "binary",
        "class_mapping": {"0": "sem_risco", "1": "em_risco"},
        "features": list(X.columns),
        "metrics": metrics,
        "cross_validation": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in cv_results.items()
        },
        "hyperparameters": _DEFAULT_HYPERPARAMS["random_forest"],
        "training_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "cross_validation_folds": 5,
    }

    metadata_path = Path(model_output_path).parent / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Metadados salvos em: {metadata_path}")

    logger.info("=== Pipeline de treinamento concluído ===")
    return model, metrics
