# Passos Mágicos ML — Predição de Risco de Defasagem Escolar

## 1. Visão Geral do Projeto

Este projeto foi desenvolvido para o **Datathon da Pós Tech — Case Passos Mágicos**. A [Associação Passos Mágicos](https://passosmagicos.org.br/) atua na transformação da vida de crianças e jovens de baixa renda por meio da educação.

O objetivo é construir um **pipeline completo de Machine Learning** capaz de estimar o risco de **defasagem escolar** de cada estudante, com base nos dados da Pesquisa Extensiva do Desenvolvimento Educacional (PEDE) dos períodos de 2022, 2023 e 2024.

A variável alvo é a **Defasagem**: valores negativos indicam risco (estudante atrasado em relação à fase ideal), enquanto valores zero ou positivos indicam situação regular.

O sistema inclui:
- Pipeline de pré-processamento e engenharia de atributos
- Treinamento e avaliação de modelos de classificação
- API REST para predições em tempo real
- Interface visual com Streamlit para demonstração e monitoramento
- Empacotamento com Docker
- Deploy na AWS (ECR + App Runner)
- Testes automatizados com cobertura mínima de 80%
- Monitoramento de drift e dashboard

**Repositório: CORRETO** [GitHub — passos-magicos-ml](https://github.com/rntavares/datathon)

**Repositório: Antigo** [AWS CodeCommit — passos-magicos-ml](https://git-codecommit.us-east-1.amazonaws.com/v1/repos/passos-magicos-ml)

## 2. Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.11 |
| ML Framework | scikit-learn 1.5 |
| API | FastAPI + Uvicorn |
| Validação | Pydantic |
| Dados | pandas, numpy, openpyxl |
| Testes | pytest, Hypothesis (property-based testing) |
| Container | Docker |
| Cloud | AWS ECR + App Runner |
| Monitoramento | scipy (teste KS para drift), logging |

## 3. Estrutura do Projeto

```
passos-magicos-ml/
├── app/                          # Aplicação FastAPI
│   ├── __init__.py
│   ├── main.py                   # Inicialização da app, middleware, lifespan
│   ├── routes.py                 # Endpoints REST e schemas Pydantic
│   └── model/                    # Artefatos do modelo treinado
│       ├── model.joblib           # Modelo serializado (gerado após treino)
│       └── metadata.json          # Metadados e métricas (gerado após treino)
├── src/                          # Pipeline de ML
│   ├── __init__.py
│   ├── preprocessing.py          # Ingestão, limpeza e normalização dos dados
│   ├── feature_engineering.py    # Criação de atributos derivados e seleção
│   ├── train.py                  # Treinamento, validação cruzada e serialização
│   ├── evaluate.py               # Métricas, matriz de confusão, threshold
│   └── utils.py                  # Logging, serialização, detecção de drift
├── monitoring/                   # Monitoramento
│   └── dashboard.py              # Gerador de dashboard HTML de monitoramento
├── tests/                        # Testes automatizados
│   ├── __init__.py
│   ├── unit/                     # Testes unitários
│   ├── integration/              # Testes de integração da API
│   └── properties/               # Testes de propriedade (Hypothesis)
├── data/                         # Dados brutos (não versionados)
│   └── BASE DE DADOS PEDE 2024 - DATATHON.xlsx
├── logs/                         # Logs de execução
├── Dockerfile                    # Imagem Docker da API
├── deploy.sh                     # Script de deploy AWS (ECR + App Runner)
├── requirements.txt              # Dependências de produção
├── requirements-dev.txt          # Dependências de desenvolvimento e testes
├── .gitignore                    # Arquivos ignorados pelo Git
└── README.md                     # Este arquivo
```

## 4. Instruções de Instalação

### Pré-requisitos

- Python 3.11+
- pip
- (Opcional) Docker para deploy containerizado

### Configuração do ambiente

```bash
# Clonar o repositório
git clone <URL_DO_REPOSITORIO>
cd passos-magicos-ml

# Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Instalar dependências de produção
pip install -r requirements.txt

# Instalar dependências de desenvolvimento (inclui pytest, hypothesis)
pip install -r requirements-dev.txt
```

### Dados

Coloque o arquivo `BASE DE DADOS PEDE 2024 - DATATHON.xlsx` no diretório `data/`.

## 5. Etapas do Pipeline de ML

O pipeline é composto por 4 etapas principais:

### 5.1 Pré-processamento (`src/preprocessing.py`)

1. **Ingestão**: Carrega as 3 abas do Excel (PEDE2022, PEDE2023, PEDE2024) e unifica em um DataFrame
2. **Limpeza**: Substitui valores `ERROR:#N/A` e `ERROR:#DIV/0!` por NaN, exclui registros de Fase ≥ 8
3. **Harmonização**: Normaliza nomes de colunas entre abas e padroniza Gênero (Menino→Masculino, Menina→Feminino)
4. **Imputação**: Mediana para indicadores PEDE e notas, moda para categóricas, remoção de colunas com >50% ausentes
5. **Encoding**: Label Encoding binário (Gênero, Indicado, Atingiu PV), One-Hot (Instituição de ensino), Ordinal (Pedra)
6. **Normalização**: StandardScaler ou MinMaxScaler para variáveis numéricas

### 5.2 Engenharia de Atributos (`src/feature_engineering.py`)

1. **Features temporais**: Deltas dos indicadores PEDE entre períodos para estudantes multi-período
2. **Features derivadas**: `media_notas`, `ratio_IDA_IEG`, `idade_vs_fase`, `anos_no_programa`
3. **Seleção**: Remoção de colunas não-preditivas (RA, Nome, Avaliadores, Destaque*, Fase Ideal)
4. **Colinearidade**: Remoção de features com correlação absoluta > 0.9

### 5.3 Treinamento (`src/train.py`)

1. **Classificação binária**: Defasagem < 0 → "em_risco" (1), ≥ 0 → "sem_risco" (0)
2. **Algoritmos suportados**: Random Forest, Gradient Boosting, Logistic Regression
3. **Validação cruzada**: 5 folds com métricas de accuracy, F1, precision e recall
4. **Serialização**: Modelo salvo em `app/model/model.joblib`, metadados em `app/model/metadata.json`

### 5.4 Avaliação (`src/evaluate.py`)

1. **Métricas**: Accuracy, Precision, Recall, F1-Score (média ponderada)
2. **Matriz de confusão** e **relatório de classificação** completo
3. **Threshold de qualidade**: Alerta via logging se métrica principal < limiar configurável

## 6. Como Treinar o Modelo

```bash
# Executar o pipeline completo de treinamento
python -c "
from src.preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.train import run_training_pipeline
from src.utils import setup_logging

setup_logging('INFO', 'logs/training.log')

df = run_preprocessing('data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx')
df = run_feature_engineering(df)
model, metrics = run_training_pipeline(df)
print('Métricas:', metrics)
"
```

Após o treinamento, os artefatos serão salvos em:
- `app/model/model.joblib` — modelo serializado
- `app/model/metadata.json` — metadados e métricas

## 7. Deploy Local com Docker

```bash
# Construir a imagem
docker build -t passos-magicos-ml .

# Executar o container
docker run -d -p 8000:8000 --name passos-ml passos-magicos-ml

# Verificar se a API está respondendo
curl http://localhost:8000/health
```

Para parar o container:

```bash
docker stop passos-ml
docker rm passos-ml
```

## 8. Deploy AWS (ECR + App Runner)

### Pré-requisitos

- AWS CLI configurado (`aws configure`)
- Docker instalado
- Permissões IAM para ECR e App Runner

### Usando o script de deploy

```bash
# Tornar o script executável (se necessário)
chmod +x deploy.sh

# Executar o deploy (usa us-east-1 por padrão)
./deploy.sh

# Ou com variáveis customizadas
AWS_REGION=sa-east-1 ECR_REPO=meu-repo APP_RUNNER_SERVICE=minha-api ./deploy.sh
```

O script realiza automaticamente:
1. Criação do repositório ECR (se não existir)
2. Build da imagem Docker
3. Push da imagem para o ECR
4. Criação ou atualização do serviço App Runner com health check configurado

### Health Check

O App Runner verifica o endpoint `GET /health` a cada 10 segundos para garantir que o serviço está operacional.

## 9. Exemplos de Chamadas à API

### Health Check

```bash
curl -X GET http://localhost:8000/health
```

Resposta:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Predição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "INDE": 7.5,
    "IAA": 8.0,
    "IEG": 7.2,
    "IPS": 6.8,
    "IDA": 7.0,
    "IPP": 6.5,
    "IPV": 7.8,
    "IAN": 6.0,
    "Matem": 7.5,
    "Portug": 8.0,
    "Ingles": 6.5,
    "Idade": 12,
    "Genero": "Masculino",
    "Fase": 4,
    "Pedra": "Ametista",
    "Indicado": "Sim",
    "Atingiu_PV": "Sim"
  }'
```

Resposta:
```json
{
  "prediction": 0,
  "risk_label": "sem_defasagem",
  "probability": 0.85,
  "probabilities": {
    "sem_defasagem": 0.85,
    "defasagem_leve": 0.10,
    "defasagem_severa": 0.05
  }
}
```

### Documentação interativa

Com a API em execução, acesse a documentação Swagger em:
- `http://localhost:8000/docs` — Swagger UI
- `http://localhost:8000/redoc` — ReDoc

## 10. Interface Visual com Streamlit

O projeto inclui uma aplicação Streamlit com 3 páginas interativas:

- **🎯 Predição de Risco** — Formulário visual com sliders para indicadores PEDE, resultado com card colorido e probabilidades
- **📊 Métricas do Modelo** — Cards com accuracy/precision/recall/F1, informações do modelo, validação cruzada
- **🔍 Monitoramento de Drift** — Tabela com status por feature (🟢/🟡/🔴), métricas resumo

### Executar o Streamlit

```bash
cd passos-magicos-ml
PYTHONPATH=. streamlit run streamlit_app.py
```

Acesse em `http://localhost:8501`.

## 11. Testes

### Executar todos os testes

```bash
pytest
```

### Executar com cobertura

```bash
pytest --cov=src --cov=app --cov-report=term-missing --cov-fail-under=80
```

### Executar testes por categoria

```bash
# Testes unitários
pytest tests/unit/

# Testes de integração
pytest tests/integration/

# Testes de propriedade (Hypothesis)
pytest tests/properties/
```

### Executar um teste específico

```bash
pytest tests/unit/test_preprocessing.py -v
```

## 11. Monitoramento

### Dashboard de Monitoramento

O projeto inclui um gerador de dashboard HTML que exibe:
- Métricas do modelo (accuracy, precision, recall, F1)
- Detecção de drift por feature com indicadores visuais (verde/amarelo/vermelho)
- Placeholder para volume de requisições

Para gerar o dashboard:

```bash
cd passos-magicos-ml
python monitoring/dashboard.py
```

O arquivo `monitoring/dashboard.html` será gerado e pode ser aberto em qualquer navegador.

### Detecção de Drift

O módulo `src/utils.py` implementa detecção de drift estatístico usando o teste Kolmogorov-Smirnov (KS) para features numéricas. O drift é detectado quando o p-value é inferior ao threshold configurado (padrão: 0.05).

### Logging

Todos os módulos utilizam o sistema de logging do Python com o formato padronizado:
```
[2024-01-15 10:30:00] [INFO] [passos_magicos] Mensagem de log
```

Níveis de severidade: DEBUG, INFO, WARNING, ERROR.

Os logs podem ser direcionados para arquivo configurando `setup_logging(level, log_file)`.

---

Desenvolvido para o Datathon Pós Tech — FIAP | Case Passos Mágicos
