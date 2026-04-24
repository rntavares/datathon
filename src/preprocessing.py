"""Módulo de pré-processamento do pipeline Passos Mágicos.

Responsável por carregar, limpar, codificar e normalizar os dados brutos
do Dataset Passos Mágicos (PEDE 2022-2024) para alimentar o pipeline de ML.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

logger = logging.getLogger("passos_magicos")

# Mapeamento de colunas da aba PEDE2022 para o padrão unificado.
# Em 2022 alguns nomes diferem das abas 2023/2024.
COLUMN_MAPPING_2022 = {
    "Defas": "Defasagem",
    "Idade 22": "Idade",
    "Matem": "Mat",
    "Portug": "Por",
    "Inglês": "Ing",
    "Fase ideal": "Fase Ideal",
    "Pedra 22": "Pedra",
    "INDE 22": "INDE",
    "Nome": "Nome Anonimizado",
    "Ano nasc": "Data de Nasc",
}

# Mapeamento de colunas da aba PEDE2023 para o padrão unificado.
COLUMN_MAPPING_2023 = {
    "INDE 2023": "INDE",
    "Pedra 2023": "Pedra",
    "INDE 23": "INDE_hist_23",
}

# Mapeamento de colunas da aba PEDE2024 para o padrão unificado.
COLUMN_MAPPING_2024 = {
    "INDE 2024": "INDE",
    "Pedra 2024": "Pedra",
    "INDE 23": "INDE_hist_23",
    "Escola": "Escola",
    "Ativo/Inativo": "Ativo_Inativo",
}

# Valores de Fase que devem ser excluídos (universitários com dados ERROR).
FASES_EXCLUIDAS = {"8", "9", "FASE 8", "FASE 9"}

# Mapeamento ordinal para a classificação Pedra (ordem crescente de INDE).
PEDRA_ORDINAL = {
    "Quartzo": 0,
    "Ágata": 1,
    "Agata": 1,
    "Ametista": 2,
    "Topázio": 3,
    "Topazio": 3,
}

# Colunas de indicadores PEDE (numéricas contínuas).
INDICADORES_PEDE = ["INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]

# Colunas de notas acadêmicas.
COLUNAS_NOTAS = ["Mat", "Por", "Ing"]

# Colunas categóricas binárias (Sim/Não ou Masculino/Feminino).
COLUNAS_BINARIAS = ["Gênero", "Indicado", "Atingiu PV"]

# Colunas categóricas para One-Hot Encoding.
COLUNAS_ONEHOT = ["Instituição de ensino"]


def load_data(file_path: str) -> pd.DataFrame:
    """Carrega o dataset Excel (PEDE 2024 - DATATHON) e retorna um DataFrame unificado.

    O arquivo Excel contém 3 abas: PEDE2022, PEDE2023 e PEDE2024.
    A função harmoniza os nomes de colunas entre as abas, adiciona coluna
    'ano_pede' para identificar o período, e concatena em um único DataFrame.

    Tratamentos específicos:
    - Substitui valores 'ERROR:#N/A' e 'ERROR:#DIV/0!' por NaN
    - Exclui registros de Fase >= 8 (universitários com dados majoritariamente ERROR)
    - Normaliza Gênero: Menino → Masculino, Menina → Feminino

    Args:
        file_path: Caminho para o arquivo .xlsx do dataset.

    Returns:
        DataFrame com dados brutos carregados e harmonizados.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o arquivo estiver corrompido ou em formato inválido.
    """
    logger.info(f"Carregando dataset: {file_path}")

    try:
        xlsx = pd.ExcelFile(file_path)
    except FileNotFoundError:
        logger.error(f"Arquivo não encontrado: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Erro ao abrir arquivo {file_path}: {e}")
        raise ValueError(f"Arquivo corrompido ou formato inválido: {file_path}") from e

    expected_sheets = ["PEDE2022", "PEDE2023", "PEDE2024"]
    for sheet in expected_sheets:
        if sheet not in xlsx.sheet_names:
            raise ValueError(f"Aba {sheet} não encontrada no arquivo {file_path}")

    dfs = []
    sheet_year_map = {
        "PEDE2022": (2022, COLUMN_MAPPING_2022),
        "PEDE2023": (2023, COLUMN_MAPPING_2023),
        "PEDE2024": (2024, COLUMN_MAPPING_2024),
    }

    for sheet_name, (ano, col_mapping) in sheet_year_map.items():
        df_sheet = pd.read_excel(xlsx, sheet_name=sheet_name)
        logger.info(f"Aba {sheet_name}: {len(df_sheet)} registros carregados")

        # Harmonizar nomes de colunas
        df_sheet = df_sheet.rename(columns=col_mapping)
        df_sheet["ano_pede"] = ano

        dfs.append(df_sheet)

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"DataFrame unificado: {df.shape[0]} registros, {df.shape[1]} colunas")

    # Substituir valores ERROR por NaN
    error_values = ["ERROR:#N/A", "ERROR:#DIV/0!"]
    for err_val in error_values:
        mask = df.isin([err_val])
        count = mask.sum().sum()
        if count > 0:
            logger.warning(f"{count} valores '{err_val}' substituídos por NaN")
            df = df.replace(err_val, np.nan)

    # Excluir registros de Fase >= 8
    if "Fase" in df.columns:
        fase_str = df["Fase"].astype(str).str.strip().str.upper()
        mask_excluir = fase_str.isin({f.upper() for f in FASES_EXCLUIDAS})

        # Também excluir valores numéricos >= 8
        fase_numeric = pd.to_numeric(df["Fase"], errors="coerce")
        mask_excluir = mask_excluir | (fase_numeric >= 8)

        n_excluidos = mask_excluir.sum()
        if n_excluidos > 0:
            logger.info(f"{n_excluidos} registros de Fase >= 8 excluídos")
            df = df[~mask_excluir].reset_index(drop=True)

    # Normalizar Gênero: Menino → Masculino, Menina → Feminino
    if "Gênero" in df.columns:
        genero_map = {"Menino": "Masculino", "Menina": "Feminino"}
        df["Gênero"] = df["Gênero"].replace(genero_map)
        logger.info("Gênero normalizado: Menino → Masculino, Menina → Feminino")

    logger.info(
        f"Dataset carregado com sucesso: {df.shape[0]} registros, {df.shape[1]} colunas"
    )
    return df


def handle_missing_values(
    df: pd.DataFrame, strategy: dict[str, str] | None = None
) -> pd.DataFrame:
    """Aplica estratégias de imputação para valores ausentes.

    Estratégias padrão por tipo de coluna:
    - Indicadores PEDE (INDE, IAA, IEG, IPS, IDA, IPP, IPV, IAN): mediana
    - Notas (Mat, Por, Ing): mediana
    - Categóricas (Gênero, Instituição de ensino, Pedra): moda
    - Colunas com >50% ausentes: remoção da coluna

    Args:
        df: DataFrame com possíveis valores ausentes.
        strategy: Dicionário mapeando coluna → estratégia ('mean', 'median', 'mode', 'drop').
                  Se None, usa estratégia padrão descrita acima.

    Returns:
        DataFrame com valores imputados.
    """
    df = df.copy()
    logger.info(f"Iniciando tratamento de valores ausentes. Shape: {df.shape}")

    # Remover colunas com >50% de valores ausentes
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
    if cols_to_drop:
        logger.warning(
            f"Colunas removidas (>50% ausentes): {cols_to_drop}"
        )
        df = df.drop(columns=cols_to_drop)

    if strategy is not None:
        # Aplicar estratégias customizadas
        for col, strat in strategy.items():
            if col not in df.columns:
                continue
            if strat == "mean":
                val = df[col].mean()
                count = df[col].isnull().sum()
                df[col] = df[col].fillna(val)
                if count > 0:
                    logger.info(f"Coluna '{col}': {count} valores imputados com média ({val:.4f})")
            elif strat == "median":
                val = df[col].median()
                count = df[col].isnull().sum()
                df[col] = df[col].fillna(val)
                if count > 0:
                    logger.info(f"Coluna '{col}': {count} valores imputados com mediana ({val:.4f})")
            elif strat == "mode":
                mode_vals = df[col].mode()
                if len(mode_vals) > 0:
                    val = mode_vals.iloc[0]
                    count = df[col].isnull().sum()
                    df[col] = df[col].fillna(val)
                    if count > 0:
                        logger.info(f"Coluna '{col}': {count} valores imputados com moda ({val})")
            elif strat == "drop":
                count = df[col].isnull().sum()
                if count > 0:
                    logger.info(f"Coluna '{col}' removida por estratégia 'drop'")
                df = df.drop(columns=[col])
        return df

    # Estratégia padrão
    imputed_counts: dict[str, int] = {}

    # Indicadores PEDE e notas → mediana
    numeric_cols = [
        c for c in INDICADORES_PEDE + COLUNAS_NOTAS if c in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            imputed_counts[col] = int(n_missing)

    # Rankings e outras numéricas → mediana
    other_numeric = ["Cg", "Cf", "Ct", "Nº Av", "Idade"]
    for col in other_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                imputed_counts[col] = int(n_missing)

    # Categóricas → moda
    categorical_cols = [
        c
        for c in ["Gênero", "Instituição de ensino", "Pedra", "Indicado", "Atingiu PV",
                   "Rec Av1", "Rec Av2", "Rec Av3", "Rec Av4", "Rec Psicologia"]
        if c in df.columns
    ]
    for col in categorical_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            mode_vals = df[col].mode()
            if len(mode_vals) > 0:
                df[col] = df[col].fillna(mode_vals.iloc[0])
                imputed_counts[col] = int(n_missing)

    if imputed_counts:
        logger.info(f"Valores ausentes imputados: {imputed_counts}")

    logger.info(f"Tratamento de valores ausentes concluído. Shape: {df.shape}")
    return df


def encode_categorical(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Aplica codificação para variáveis categóricas.

    Codificações específicas:
    - Gênero (Masculino/Feminino): Label Encoding binário
    - Indicado (Sim/Não): Label Encoding binário
    - Atingiu PV (Sim/Não): Label Encoding binário
    - Instituição de ensino: One-Hot Encoding
    - Pedra (Quartzo/Ágata/Ametista/Topázio): Ordinal Encoding (ordem crescente de INDE)

    Args:
        df: DataFrame com colunas categóricas.
        columns: Lista de colunas a codificar. Se None, detecta automaticamente.

    Returns:
        DataFrame com variáveis categóricas codificadas.
    """
    df = df.copy()
    logger.info("Iniciando codificação de variáveis categóricas")

    if columns is not None:
        cols_to_encode = [c for c in columns if c in df.columns]
    else:
        cols_to_encode = None

    # Label Encoding binário para Gênero
    if "Gênero" in df.columns and (cols_to_encode is None or "Gênero" in cols_to_encode):
        genero_map = {"Masculino": 1, "Feminino": 0}
        df["Gênero"] = df["Gênero"].map(genero_map)
        # Preencher valores não mapeados com 0
        df["Gênero"] = df["Gênero"].fillna(0).astype(int)
        logger.info("Gênero codificado: Masculino=1, Feminino=0")

    # Label Encoding binário para Indicado
    if "Indicado" in df.columns and (cols_to_encode is None or "Indicado" in cols_to_encode):
        indicado_map = {"Sim": 1, "Não": 0, "sim": 1, "não": 0}
        df["Indicado"] = df["Indicado"].map(indicado_map)
        df["Indicado"] = df["Indicado"].fillna(0).astype(int)
        logger.info("Indicado codificado: Sim=1, Não=0")

    # Label Encoding binário para Atingiu PV
    if "Atingiu PV" in df.columns and (cols_to_encode is None or "Atingiu PV" in cols_to_encode):
        atingiu_map = {"Sim": 1, "Não": 0, "sim": 1, "não": 0}
        df["Atingiu PV"] = df["Atingiu PV"].map(atingiu_map)
        df["Atingiu PV"] = df["Atingiu PV"].fillna(0).astype(int)
        logger.info("Atingiu PV codificado: Sim=1, Não=0")

    # Ordinal Encoding para Pedra
    if "Pedra" in df.columns and (cols_to_encode is None or "Pedra" in cols_to_encode):
        df["Pedra"] = df["Pedra"].map(PEDRA_ORDINAL)
        df["Pedra"] = df["Pedra"].fillna(-1).astype(int)
        logger.info(
            "Pedra codificada ordinalmente: Quartzo=0, Ágata=1, Ametista=2, Topázio=3"
        )

    # One-Hot Encoding para Instituição de ensino
    if "Instituição de ensino" in df.columns and (
        cols_to_encode is None or "Instituição de ensino" in cols_to_encode
    ):
        df["Instituição de ensino"] = df["Instituição de ensino"].fillna("Desconhecida")
        dummies = pd.get_dummies(
            df["Instituição de ensino"], prefix="Inst", dtype=int
        )
        df = pd.concat([df.drop(columns=["Instituição de ensino"]), dummies], axis=1)
        logger.info(
            f"Instituição de ensino codificada via One-Hot: {list(dummies.columns)}"
        )

    # Ordinal Encoding para colunas Rec Av (recomendações de avaliadores)
    rec_av_cols = [c for c in df.columns if c.startswith("Rec Av")]
    rec_av_to_encode = [
        c for c in rec_av_cols
        if cols_to_encode is None or c in cols_to_encode
    ]
    if rec_av_to_encode:
        rec_av_map = {
            "Promovido de Fase": 2,
            "Mantido na Fase atual": 1,
            "Mantido na fase atual": 1,
        }
        for col in rec_av_to_encode:
            if df[col].dtype == object:
                df[col] = df[col].map(rec_av_map)
                df[col] = df[col].fillna(0).astype(int)
        logger.info(f"Rec Av codificadas ordinalmente: {rec_av_to_encode}")

    logger.info(f"Codificação categórica concluída. Shape: {df.shape}")
    return df


def normalize_numerical(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "standard",
) -> pd.DataFrame:
    """Aplica normalização/padronização em variáveis numéricas.

    Colunas numéricas do dataset PEDE:
    - Indicadores: INDE, IAA, IEG, IPS, IDA, IPP, IPV, IAN (escala ~0-10)
    - Notas: Mat, Por, Ing (escala ~0-10)
    - Rankings: Cg, Cf, Ct (inteiros positivos)
    - Idade: inteiro
    - Nº Av: número de avaliações (inteiro)

    Args:
        df: DataFrame com colunas numéricas.
        columns: Lista de colunas a normalizar. Se None, detecta automaticamente.
        method: 'standard' (StandardScaler) ou 'minmax' (MinMaxScaler).

    Returns:
        DataFrame com variáveis numéricas normalizadas.

    Raises:
        ValueError: Se o método informado não for 'standard' nem 'minmax'.
    """
    df = df.copy()
    logger.info(f"Iniciando normalização numérica (método: {method})")

    if method not in ("standard", "minmax"):
        raise ValueError(f"Método de normalização inválido: '{method}'. Use 'standard' ou 'minmax'.")

    if columns is not None:
        num_cols = [c for c in columns if c in df.columns]
    else:
        # Detectar colunas numéricas automaticamente
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Excluir coluna alvo e identificadores do scaling
        exclude = ["ano_pede", "Defasagem"]
        num_cols = [c for c in num_cols if c not in exclude]

    if not num_cols:
        logger.warning("Nenhuma coluna numérica encontrada para normalização")
        return df

    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df[num_cols] = scaler.fit_transform(df[num_cols])
    logger.info(f"Colunas normalizadas ({method}): {num_cols}")
    logger.info(f"Normalização concluída. Shape: {df.shape}")
    return df


def run_preprocessing(file_path: str) -> pd.DataFrame:
    """Executa o pipeline completo de pré-processamento.

    Etapas:
    1. Carregar e harmonizar dados das 3 abas do Excel
    2. Tratar valores ausentes
    3. Codificar variáveis categóricas
    4. Normalizar variáveis numéricas

    Args:
        file_path: Caminho para o arquivo .xlsx.

    Returns:
        DataFrame limpo e pré-processado.
    """
    logger.info("=== Iniciando pipeline de pré-processamento ===")

    df = load_data(file_path)
    logger.info(f"Etapa 1/4 concluída — Dados carregados: {df.shape}")

    df = handle_missing_values(df)
    logger.info(f"Etapa 2/4 concluída — Valores ausentes tratados: {df.shape}")

    df = encode_categorical(df)
    logger.info(f"Etapa 3/4 concluída — Categóricas codificadas: {df.shape}")

    df = normalize_numerical(df)
    logger.info(f"Etapa 4/4 concluída — Numéricas normalizadas: {df.shape}")

    logger.info(
        f"=== Pipeline de pré-processamento concluído: {df.shape[0]} registros, "
        f"{df.shape[1]} colunas ==="
    )
    return df
