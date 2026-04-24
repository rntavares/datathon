"""Aplicação principal FastAPI — Passos Mágicos ML API.

Inicializa a aplicação, carrega o modelo serializado na inicialização
e registra rotas, middleware de logging e tratamento global de exceções.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routes import router
from src.utils import load_model

logger = logging.getLogger("passos_magicos")

MODEL_PATH = "app/model/model.joblib"


# ---------------------------------------------------------------------------
# Lifespan — carregamento do modelo na inicialização
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo ao iniciar e libera recursos ao encerrar."""
    try:
        app.state.model = load_model(MODEL_PATH)
        logger.info("Modelo carregado com sucesso na inicialização da API")
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(f"Modelo não encontrado ou inválido ({exc}). API iniciará sem modelo.")
        app.state.model = None

    yield  # aplicação em execução

    # Cleanup (se necessário no futuro)
    logger.info("Encerrando aplicação Passos Mágicos ML API")


# ---------------------------------------------------------------------------
# Instância FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Passos Mágicos ML API",
    description=(
        "API de predição de risco de defasagem escolar para estudantes "
        "da Associação Passos Mágicos, baseada em dados PEDE 2022-2024."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


# ---------------------------------------------------------------------------
# Middleware — logging de requisições
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Registra timestamp, endpoint e tempo de resposta de cada requisição."""
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    logger.info(
        f"{request.method} {request.url.path} — "
        f"status={response.status_code} — "
        f"tempo={duration_ms:.1f}ms"
    )
    return response


# ---------------------------------------------------------------------------
# Exception handler global
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Captura exceções não tratadas e retorna 500 genérico."""
    logger.error(f"Erro interno: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno do servidor"},
    )
