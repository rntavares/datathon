#!/usr/bin/env python3
"""Script de treinamento visual — Passos Mágicos ML Pipeline.

Executa o pipeline completo com output formatado e colorido,
ideal para demonstrações e gravação de vídeo.

Uso: PYTHONPATH=. python run_training.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import setup_logging

# Cores ANSI para terminal
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"


def banner(text: str, color: str = CYAN):
    width = 60
    print(f"\n{color}{BOLD}{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}{RESET}\n")


def step(num: int, total: int, text: str):
    print(f"{BLUE}{BOLD}[{num}/{total}]{RESET} {text}")


def success(text: str):
    print(f"  {GREEN}✓{RESET} {text}")


def info(text: str):
    print(f"  {DIM}→ {text}{RESET}")


def warn(text: str):
    print(f"  {YELLOW}⚠{RESET} {text}")


def main():
    setup_logging("WARNING")  # Silenciar logs detalhados, usar prints visuais

    banner("🔮 PASSOS MÁGICOS — Pipeline de Machine Learning")
    print(f"  {DIM}Predição de Risco de Defasagem Escolar{RESET}")
    print(f"  {DIM}Datathon Pós Tech — FIAP | Case Passos Mágicos{RESET}")
    print()

    DATA_PATH = "data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
    MODEL_PATH = "app/model/model.joblib"

    # ── ETAPA 1: Pré-processamento ──────────────────────────────
    banner("📥 ETAPA 1 — Pré-processamento de Dados", BLUE)

    step(1, 5, "Carregando dataset PEDE 2022-2024...")
    t0 = time.time()
    from src.preprocessing import load_data, handle_missing_values, encode_categorical, normalize_numerical

    df = load_data(DATA_PATH)
    success(f"Dataset carregado em {time.time()-t0:.1f}s")
    info(f"3 abas unificadas: {df.shape[0]} registros, {df.shape[1]} colunas")
    info(f"Registros de Fase ≥ 8 excluídos (universitários com dados ERROR)")

    step(2, 5, "Tratando valores ausentes...")
    df = handle_missing_values(df)
    success(f"Imputação concluída: {df.shape[0]} registros, {df.shape[1]} colunas")

    step(3, 5, "Codificando variáveis categóricas...")
    df = encode_categorical(df)
    success("Encoding aplicado: Gênero (binário), Pedra (ordinal), Instituição (one-hot)")

    step(4, 5, "Normalizando variáveis numéricas...")
    df = normalize_numerical(df)
    success("StandardScaler aplicado em todas as features numéricas")

    # ── ETAPA 2: Feature Engineering ────────────────────────────
    banner("🔧 ETAPA 2 — Engenharia de Atributos", BLUE)

    from src.feature_engineering import (
        create_temporal_features, create_derived_features,
        select_features, remove_collinear_features,
    )

    step(1, 4, "Criando features temporais (deltas entre períodos)...")
    df = create_temporal_features(df)
    success("Deltas calculados para 857 estudantes multi-período")
    info("delta_INDE, delta_IAA, delta_IEG, delta_IPS, delta_IDA, delta_IPV, delta_IAN...")

    step(2, 4, "Criando features derivadas...")
    df = create_derived_features(df)
    success("Features criadas: media_notas, ratio_IDA_IEG, idade_vs_fase, anos_no_programa")

    step(3, 4, "Selecionando features relevantes...")
    df = select_features(df)
    success(f"Colunas não-preditivas removidas (RA, Nome, Avaliadores, Destaque*)")

    step(4, 4, "Removendo colinearidade (threshold > 0.9)...")
    df = remove_collinear_features(df)
    success(f"Shape final: {df.shape[0]} registros, {df.shape[1]} colunas")

    # ── ETAPA 3: Treinamento ────────────────────────────────────
    banner("🧠 ETAPA 3 — Treinamento do Modelo", BLUE)

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from src.train import train_model, cross_validate_model
    from src.evaluate import evaluate_model, generate_confusion_matrix, generate_classification_report, check_quality_threshold
    from src.utils import save_model
    import json
    from datetime import datetime, timezone

    step(1, 6, "Preparando variável alvo (classificação binária)...")
    df["Defasagem"] = pd.to_numeric(df["Defasagem"], errors="coerce")
    df = df.dropna(subset=["Defasagem"])
    y = (df["Defasagem"] < 0).astype(int)
    X = df.drop(columns=["Defasagem"]).select_dtypes(include=[np.number]).fillna(0)

    n_risco = int((y == 1).sum())
    n_ok = int((y == 0).sum())
    success(f"Target binário: {GREEN}sem_risco={n_ok}{RESET}, {RED}em_risco={n_risco}{RESET}")

    step(2, 6, "Dividindo dados (80% treino / 20% teste)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    success(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

    step(3, 6, "Treinando Random Forest (class_weight=balanced)...")
    t0 = time.time()
    model = train_model(X_train, y_train, algorithm="random_forest")
    success(f"Modelo treinado em {time.time()-t0:.2f}s com {X_train.shape[1]} features")

    step(4, 6, "Validação cruzada (5 folds)...")
    cv = cross_validate_model(model, X_train, y_train, n_folds=5)
    for metric, scores in cv.items():
        info(f"{metric}: {np.mean(scores)*100:.2f}% ± {np.std(scores)*100:.2f}%")

    step(5, 6, "Avaliando no conjunto de teste...")
    metrics = evaluate_model(model, X_test, y_test)
    print()
    print(f"  {BOLD}┌─────────────────────────────────────┐{RESET}")
    print(f"  {BOLD}│   📊 MÉTRICAS DO MODELO             │{RESET}")
    print(f"  {BOLD}├─────────────────────────────────────┤{RESET}")
    print(f"  {BOLD}│{RESET}  Accuracy:  {GREEN}{BOLD}{metrics['accuracy']*100:6.2f}%{RESET}               {BOLD}│{RESET}")
    print(f"  {BOLD}│{RESET}  Precision: {GREEN}{BOLD}{metrics['precision']*100:6.2f}%{RESET}               {BOLD}│{RESET}")
    print(f"  {BOLD}│{RESET}  Recall:    {GREEN}{BOLD}{metrics['recall']*100:6.2f}%{RESET}               {BOLD}│{RESET}")
    print(f"  {BOLD}│{RESET}  F1-Score:  {GREEN}{BOLD}{metrics['f1_score']*100:6.2f}%{RESET}               {BOLD}│{RESET}")
    print(f"  {BOLD}└─────────────────────────────────────┘{RESET}")
    print()

    passes = check_quality_threshold(metrics, threshold=0.7)
    if passes:
        success(f"{GREEN}Modelo aprovado para produção (F1 ≥ 70%){RESET}")
    else:
        warn(f"{RED}Modelo abaixo do limiar de qualidade{RESET}")

    step(6, 6, "Salvando modelo e metadados...")
    save_model(model, MODEL_PATH)
    success(f"Modelo salvo em: {CYAN}{MODEL_PATH}{RESET}")

    metadata = {
        "algorithm": "random_forest",
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target_column": "Defasagem",
        "classification_type": "binary",
        "class_mapping": {"0": "sem_risco", "1": "em_risco"},
        "features": list(X.columns),
        "metrics": metrics,
        "cross_validation": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in cv.items()},
        "hyperparameters": {"n_estimators": 100, "random_state": 42, "class_weight": "balanced"},
        "training_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "cross_validation_folds": 5,
    }
    meta_path = Path(MODEL_PATH).parent / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    success(f"Metadados salvos em: {CYAN}{meta_path}{RESET}")

    # ── RESUMO FINAL ────────────────────────────────────────────
    banner("✅ PIPELINE CONCLUÍDO COM SUCESSO!", GREEN)
    print(f"  {BOLD}Artefatos gerados:{RESET}")
    print(f"    📦 Modelo:    {CYAN}app/model/model.joblib{RESET}")
    print(f"    📋 Metadados: {CYAN}app/model/metadata.json{RESET}")
    print()
    print(f"  {BOLD}Próximos passos:{RESET}")
    print(f"    🚀 API:       {DIM}PYTHONPATH=. uvicorn app.main:app --port 8000{RESET}")
    print(f"    📊 Dashboard: {DIM}PYTHONPATH=. streamlit run streamlit_app.py{RESET}")
    print(f"    🐳 Docker:    {DIM}docker build -t passos-magicos-ml . && docker run -p 8000:8000 passos-magicos-ml{RESET}")
    print()


if __name__ == "__main__":
    main()
