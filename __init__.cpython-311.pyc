"""Script de geração do dashboard de monitoramento — Passos Mágicos ML.

Gera um arquivo HTML estático (monitoring/dashboard.html) com:
- Métricas do modelo (accuracy, precision, recall, F1) lidas de app/model/metadata.json
- Resultados de detecção de drift por feature (verde/amarelo/vermelho)
- Placeholder de volume de requisições
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("passos_magicos")

# Caminhos padrão
_METADATA_PATH = os.path.join("app", "model", "metadata.json")
_OUTPUT_PATH = os.path.join("monitoring", "dashboard.html")


def _load_metadata(path: str = _METADATA_PATH) -> dict:
    """Carrega metadados do modelo treinado."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Arquivo de metadados não encontrado: {path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Erro ao decodificar metadados: {e}")
        return {}


def _generate_sample_drift_results() -> dict[str, dict]:
    """Gera resultados de detecção de drift usando dados de amostra.

    Cria dois DataFrames sintéticos (referência e atual) e executa
    detect_drift do módulo src.utils para obter resultados reais.
    """
    try:
        from src.utils import detect_drift
    except ImportError:
        logger.warning("Não foi possível importar src.utils.detect_drift")
        return {}

    np.random.seed(42)
    n_samples = 200
    features = ["INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]

    # Dados de referência (distribuição original)
    ref_data = pd.DataFrame(
        {feat: np.random.normal(loc=5.0, scale=1.5, size=n_samples) for feat in features}
    )

    # Dados atuais (com leve drift em algumas features)
    cur_data = pd.DataFrame(
        {feat: np.random.normal(loc=5.0, scale=1.5, size=n_samples) for feat in features}
    )
    # Introduzir drift artificial em IDA e IPV
    cur_data["IDA"] = np.random.normal(loc=6.5, scale=2.0, size=n_samples)
    cur_data["IPV"] = np.random.normal(loc=3.5, scale=1.0, size=n_samples)

    return detect_drift(ref_data, cur_data, threshold=0.05)


def _drift_color(result: dict) -> str:
    """Retorna a cor do indicador de drift baseado no p-value."""
    p = result.get("p_value", 1.0)
    if result.get("drift_detected", False):
        return "#e74c3c"  # vermelho
    elif p < 0.1:
        return "#f39c12"  # amarelo (alerta)
    else:
        return "#27ae60"  # verde


def _drift_label(result: dict) -> str:
    """Retorna o rótulo textual do status de drift."""
    p = result.get("p_value", 1.0)
    if result.get("drift_detected", False):
        return "Drift Detectado"
    elif p < 0.1:
        return "Atenção"
    else:
        return "Normal"


def _build_html(metadata: dict, drift_results: dict) -> str:
    """Constrói o conteúdo HTML do dashboard."""
    metrics = metadata.get("metrics", {})
    trained_at = metadata.get("trained_at", "N/A")
    algorithm = metadata.get("algorithm", "N/A")
    version = metadata.get("version", "N/A")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Métricas do modelo
    accuracy = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    f1 = metrics.get("f1_score", 0)

    # Linhas da tabela de drift
    drift_rows = ""
    for feature, result in sorted(drift_results.items()):
        color = _drift_color(result)
        label = _drift_label(result)
        p_value = result.get("p_value", 0)
        statistic = result.get("statistic", 0)
        drift_rows += f"""
        <tr>
            <td>{feature}</td>
            <td>{p_value:.4f}</td>
            <td>{statistic:.4f}</td>
            <td>
                <span style="display:inline-block;width:14px;height:14px;
                    border-radius:50%;background:{color};margin-right:6px;
                    vertical-align:middle;"></span>
                {label}
            </td>
        </tr>"""

    if not drift_rows:
        drift_rows = '<tr><td colspan="4">Nenhum resultado de drift disponível</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard de Monitoramento — Passos Mágicos ML</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f4f6f9;
            color: #333;
            padding: 24px;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 8px;
            font-size: 1.8rem;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 32px;
            font-size: 0.95rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .card {{
            background: #fff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
            text-align: center;
        }}
        .card .label {{
            font-size: 0.85rem;
            color: #7f8c8d;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: #2c3e50;
        }}
        .section {{
            background: #fff;
            border-radius: 10px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
            margin-bottom: 24px;
        }}
        .section h2 {{
            color: #2c3e50;
            margin-bottom: 16px;
            font-size: 1.2rem;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }}
        tr:hover {{ background: #f8f9fa; }}
        .info-row {{
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }}
        .info-item {{
            font-size: 0.9rem;
            color: #555;
        }}
        .info-item strong {{ color: #2c3e50; }}
        .placeholder-box {{
            background: #f8f9fa;
            border: 2px dashed #bdc3c7;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            color: #95a5a6;
            font-size: 0.95rem;
        }}
        footer {{
            text-align: center;
            color: #95a5a6;
            font-size: 0.8rem;
            margin-top: 32px;
        }}
    </style>
</head>
<body>
    <h1>📊 Dashboard de Monitoramento</h1>
    <p class="subtitle">Passos Mágicos ML — Predição de Risco de Defasagem Escolar</p>

    <!-- Métricas do Modelo -->
    <div class="grid">
        <div class="card">
            <div class="label">Accuracy</div>
            <div class="value">{accuracy:.2%}</div>
        </div>
        <div class="card">
            <div class="label">Precision</div>
            <div class="value">{precision:.2%}</div>
        </div>
        <div class="card">
            <div class="label">Recall</div>
            <div class="value">{recall:.2%}</div>
        </div>
        <div class="card">
            <div class="label">F1-Score</div>
            <div class="value">{f1:.2%}</div>
        </div>
    </div>

    <!-- Informações do Modelo -->
    <div class="section">
        <h2>Informações do Modelo</h2>
        <div class="info-row">
            <div class="info-item"><strong>Algoritmo:</strong> {algorithm}</div>
            <div class="info-item"><strong>Versão:</strong> {version}</div>
            <div class="info-item"><strong>Treinado em:</strong> {trained_at}</div>
        </div>
    </div>

    <!-- Detecção de Drift -->
    <div class="section">
        <h2>Detecção de Drift por Feature</h2>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>p-value (KS)</th>
                    <th>Estatística KS</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {drift_rows}
            </tbody>
        </table>
    </div>

    <!-- Volume de Requisições (placeholder) -->
    <div class="section">
        <h2>Volume de Requisições</h2>
        <div class="placeholder-box">
            📈 Gráfico de volume de requisições será exibido aqui quando a API
            estiver em produção com logging de métricas habilitado.
        </div>
    </div>

    <footer>
        Gerado em {generated_at} | Passos Mágicos ML Pipeline v{version}
    </footer>
</body>
</html>"""
    return html


def generate_dashboard(
    metadata_path: str = _METADATA_PATH,
    output_path: str = _OUTPUT_PATH,
) -> str:
    """Gera o dashboard HTML de monitoramento.

    Args:
        metadata_path: Caminho para o arquivo metadata.json do modelo.
        output_path: Caminho de saída para o arquivo HTML gerado.

    Returns:
        Caminho absoluto do arquivo HTML gerado.
    """
    logger.info("Gerando dashboard de monitoramento")

    metadata = _load_metadata(metadata_path)
    drift_results = _generate_sample_drift_results()

    html_content = _build_html(metadata, drift_results)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_content, encoding="utf-8")

    abs_path = str(output.resolve())
    logger.info(f"Dashboard gerado com sucesso: {abs_path}")
    return abs_path


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    )

    path = generate_dashboard()
    print(f"Dashboard gerado: {path}")
