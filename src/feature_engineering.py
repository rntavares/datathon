"""Módulo de engenharia de atributos do pipeline Passos Mágicos.

Responsável por criar atributos derivados, calcular variações temporais
dos indicadores PEDE entre períodos, remover colinearidade e selecionar
features relevantes para o modelo de classificação.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("passos_magicos")

# Indicadores PEDE para cálculo de deltas temporais.
_INDICADORES_DELTA = ["INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]

# Colunas de notas para cálculo de deltas temporais.
_NOTAS_DELTA = ["Mat", "Por", "Ing"]

# Colunas não-preditivas a serem removidas na seleção de features.
_COLUNAS_NAO_PREDITIVAS = [
    "RA",
    "Nome Anonimizado",
    "Avaliador1",
    "Avaliador2",
    "Avaliador3",
    "Avaliador4",
    "Avaliador5",
    "Avaliador6",
    "Turma",
    "Destaque IEG",
    "Destaque IDA",
    "Destaque IPV",
    "Fase Ideal",
    "Data de Nasc",
    "Rec Av1",
    "Rec Av2",
    "Rec Av3",
    "Rec Av4",
    "Rec Psicologia",
    "Pedra 20",
    "Pedra 21",
    "Pedra 22",
    "Pedra 23",
    "INDE_hist_23",
    "ano_pede",
    "Escola",
    "Ativo_Inativo",
]


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula variações temporais dos indicadores PEDE entre períodos.

    Para cada estudante (identificado por RA) presente em múltiplos anos
    (coluna ano_pede), calcula:
    - Deltas dos indicadores PEDE entre períodos consecutivos
      (delta_INDE, delta_IAA, delta_IEG, delta_IPS, delta_IDA,
       delta_IPP, delta_IPV, delta_IAN)
    - Deltas das notas (delta_Mat, delta_Por, delta_Ing)
    - Evolução da Pedra entre períodos (pedra_evolucao)
    - Anos de permanência no programa (anos_no_programa)

    Args:
        df: DataFrame pré-processado com coluna 'ano_pede' e dados multi-período.

    Returns:
        DataFrame com atributos temporais adicionados.
    """
    df = df.copy()
    logger.info("Iniciando criação de features temporais")

    # Calcular anos no programa se possível
    if "Ano ingresso" in df.columns and "ano_pede" in df.columns:
        df["Ano ingresso"] = pd.to_numeric(df["Ano ingresso"], errors="coerce")
        df["anos_no_programa"] = df["ano_pede"] - df["Ano ingresso"]
        logger.info("Feature 'anos_no_programa' criada (ano_pede - Ano ingresso)")

    # Verificar se há dados multi-período
    if "RA" not in df.columns or "ano_pede" not in df.columns:
        logger.warning(
            "Colunas 'RA' ou 'ano_pede' ausentes — features temporais não geradas"
        )
        return df

    # Identificar estudantes presentes em múltiplos anos
    ra_counts = df.groupby("RA")["ano_pede"].nunique()
    multi_period_ras = ra_counts[ra_counts > 1].index

    if len(multi_period_ras) == 0:
        logger.warning(
            "Nenhum estudante presente em múltiplos períodos — "
            "deltas temporais não calculados"
        )
        return df

    logger.info(
        f"{len(multi_period_ras)} estudantes presentes em múltiplos períodos"
    )

    # Inicializar colunas de delta com NaN
    delta_cols_indicators = [f"delta_{col}" for col in _INDICADORES_DELTA]
    delta_cols_notas = [f"delta_{col}" for col in _NOTAS_DELTA]
    all_delta_cols = delta_cols_indicators + delta_cols_notas + ["pedra_evolucao"]

    for col in all_delta_cols:
        df[col] = np.nan

    # Calcular deltas para cada estudante multi-período
    for ra in multi_period_ras:
        mask = df["RA"] == ra
        student_data = df.loc[mask].sort_values("ano_pede")

        if len(student_data) < 2:
            continue

        # Usar o último e o penúltimo período para calcular o delta
        last_row_idx = student_data.index[-1]
        prev_row = student_data.iloc[-2]
        last_row = student_data.iloc[-1]

        # Deltas dos indicadores PEDE
        for indicator in _INDICADORES_DELTA:
            if indicator in df.columns:
                val_last = pd.to_numeric(last_row.get(indicator), errors="coerce")
                val_prev = pd.to_numeric(prev_row.get(indicator), errors="coerce")
                if pd.notna(val_last) and pd.notna(val_prev):
                    df.loc[last_row_idx, f"delta_{indicator}"] = val_last - val_prev

        # Deltas das notas
        for nota in _NOTAS_DELTA:
            if nota in df.columns:
                val_last = pd.to_numeric(last_row.get(nota), errors="coerce")
                val_prev = pd.to_numeric(prev_row.get(nota), errors="coerce")
                if pd.notna(val_last) and pd.notna(val_prev):
                    df.loc[last_row_idx, f"delta_{nota}"] = val_last - val_prev

        # Evolução da Pedra (diferença ordinal)
        if "Pedra" in df.columns:
            pedra_last = last_row.get("Pedra")
            pedra_prev = prev_row.get("Pedra")
            if pd.notna(pedra_last) and pd.notna(pedra_prev):
                pedra_last_num = pd.to_numeric(pedra_last, errors="coerce")
                pedra_prev_num = pd.to_numeric(pedra_prev, errors="coerce")
                if pd.notna(pedra_last_num) and pd.notna(pedra_prev_num):
                    df.loc[last_row_idx, "pedra_evolucao"] = (
                        pedra_last_num - pedra_prev_num
                    )

    n_deltas = df[delta_cols_indicators + delta_cols_notas].notna().any(axis=1).sum()
    logger.info(
        f"Features temporais criadas: {len(all_delta_cols)} colunas, "
        f"{n_deltas} registros com deltas calculados"
    )
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria atributos derivados a partir das colunas existentes do PEDE.

    Features derivadas:
    - media_notas: média aritmética de Mat, Por e Ing
    - ratio_IDA_IEG: razão entre indicador de aprendizagem (IDA) e engajamento (IEG)
    - idade_vs_fase: diferença entre Idade e Fase (proxy numérico de defasagem)
    - anos_no_programa: diferença entre ano_pede e Ano ingresso (se não existir)

    Args:
        df: DataFrame pré-processado.

    Returns:
        DataFrame com atributos derivados adicionados.
    """
    df = df.copy()
    logger.info("Iniciando criação de features derivadas")

    features_criadas: list[str] = []

    # Média das notas
    notas_presentes = [c for c in ["Mat", "Por", "Ing"] if c in df.columns]
    if notas_presentes:
        for col in notas_presentes:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["media_notas"] = df[notas_presentes].mean(axis=1)
        features_criadas.append("media_notas")

    # Razão IDA / IEG
    if "IDA" in df.columns and "IEG" in df.columns:
        df["IDA"] = pd.to_numeric(df["IDA"], errors="coerce")
        df["IEG"] = pd.to_numeric(df["IEG"], errors="coerce")
        # Evitar divisão por zero: substituir IEG == 0 por NaN no cálculo
        df["ratio_IDA_IEG"] = df["IDA"] / df["IEG"].replace(0, np.nan)
        features_criadas.append("ratio_IDA_IEG")

    # Idade vs Fase (proxy numérico)
    if "Idade" in df.columns and "Fase" in df.columns:
        df["Idade"] = pd.to_numeric(df["Idade"], errors="coerce")
        df["Fase"] = pd.to_numeric(df["Fase"], errors="coerce")
        df["idade_vs_fase"] = df["Idade"] - df["Fase"]
        features_criadas.append("idade_vs_fase")

    # Anos no programa (se ainda não existir)
    if "anos_no_programa" not in df.columns:
        if "Ano ingresso" in df.columns and "ano_pede" in df.columns:
            df["Ano ingresso"] = pd.to_numeric(df["Ano ingresso"], errors="coerce")
            df["anos_no_programa"] = df["ano_pede"] - df["Ano ingresso"]
            features_criadas.append("anos_no_programa")

    logger.info(f"Features derivadas criadas: {features_criadas}")
    logger.info(f"Shape após features derivadas: {df.shape}")
    return df


def remove_collinear_features(
    df: pd.DataFrame, threshold: float = 0.9
) -> pd.DataFrame:
    """Remove features com alta colinearidade.

    Calcula a matriz de correlação absoluta entre todas as colunas numéricas
    e remove a segunda coluna de cada par com correlação acima do threshold.
    A primeira coluna de cada par correlacionado é mantida.

    Nota: INDE é calculado como média ponderada dos sub-indicadores
    (IAA, IEG, IPS, IDA, IPP, IPV, IAN), portanto terá alta correlação
    com eles.

    Args:
        df: DataFrame com features candidatas.
        threshold: Limiar de correlação absoluta para remoção (padrão: 0.9).

    Returns:
        DataFrame sem features redundantes.
    """
    df = df.copy()
    logger.info(
        f"Iniciando remoção de colinearidade (threshold={threshold}). "
        f"Shape: {df.shape}"
    )

    # Selecionar apenas colunas numéricas
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        logger.info("Menos de 2 colunas numéricas — nada a remover")
        return df

    corr_matrix = numeric_df.corr().abs()

    # Máscara triangular superior (sem a diagonal)
    upper_triangle = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)

    # Identificar colunas a remover
    cols_to_drop: set[str] = set()
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            if upper_triangle[i, j] and corr_matrix.iloc[i, j] > threshold:
                col_to_remove = corr_matrix.columns[j]
                cols_to_drop.add(col_to_remove)

    if cols_to_drop:
        logger.info(
            f"Colunas removidas por colinearidade (>{threshold}): "
            f"{sorted(cols_to_drop)}"
        )
        df = df.drop(columns=list(cols_to_drop))
    else:
        logger.info("Nenhuma coluna removida por colinearidade")

    logger.info(f"Shape após remoção de colinearidade: {df.shape}")
    return df


def select_features(
    df: pd.DataFrame, target_column: str = "Defasagem"
) -> pd.DataFrame:
    """Seleciona features finais para treinamento.

    Remove colunas não-preditivas como identificadores (RA, Nome),
    metadados de avaliadores, colunas de texto livre (Destaque*),
    colunas históricas de Pedra e outras colunas auxiliares.
    Mantém a coluna alvo (target_column).

    Args:
        df: DataFrame com todas as features candidatas.
        target_column: Nome da coluna alvo (padrão: 'Defasagem').

    Returns:
        DataFrame contendo apenas features selecionadas e a coluna alvo.
    """
    df = df.copy()
    logger.info(f"Iniciando seleção de features. Shape inicial: {df.shape}")

    # Colunas a remover (presentes no DataFrame)
    cols_to_drop = [
        c for c in _COLUNAS_NAO_PREDITIVAS if c in df.columns
    ]

    # Também remover colunas que começam com "Rec Av" (variações como Rec Av5, etc.)
    rec_av_cols = [c for c in df.columns if c.startswith("Rec Av")]
    cols_to_drop.extend([c for c in rec_av_cols if c not in cols_to_drop])

    # Garantir que a coluna alvo NÃO seja removida
    cols_to_drop = [c for c in cols_to_drop if c != target_column]

    if cols_to_drop:
        logger.info(f"Colunas não-preditivas removidas: {sorted(cols_to_drop)}")
        df = df.drop(columns=cols_to_drop)

    logger.info(
        f"Seleção de features concluída. Shape final: {df.shape}. "
        f"Colunas: {list(df.columns)}"
    )
    return df


def run_feature_engineering(
    df: pd.DataFrame, target_column: str = "Defasagem"
) -> pd.DataFrame:
    """Executa o pipeline completo de engenharia de atributos.

    Etapas:
    1. Criar features temporais (deltas entre períodos)
    2. Criar features derivadas (médias, razões, proxies)
    3. Selecionar features relevantes (remover não-preditivas)
    4. Remover colinearidade

    Args:
        df: DataFrame pré-processado.
        target_column: Nome da coluna alvo (padrão: 'Defasagem').

    Returns:
        DataFrame com features finais para treinamento.
    """
    logger.info("=== Iniciando pipeline de engenharia de atributos ===")

    df = create_temporal_features(df)
    logger.info(f"Etapa 1/4 concluída — Features temporais: {df.shape}")

    df = create_derived_features(df)
    logger.info(f"Etapa 2/4 concluída — Features derivadas: {df.shape}")

    df = select_features(df, target_column=target_column)
    logger.info(f"Etapa 3/4 concluída — Features selecionadas: {df.shape}")

    df = remove_collinear_features(df)
    logger.info(f"Etapa 4/4 concluída — Colinearidade removida: {df.shape}")

    logger.info(
        f"=== Pipeline de engenharia de atributos concluído: "
        f"{df.shape[0]} registros, {df.shape[1]} colunas ==="
    )
    return df
