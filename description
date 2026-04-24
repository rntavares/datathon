#!/usr/bin/env bash
# =============================================================================
# Script de deploy — Passos Mágicos ML API
#
# Realiza o build da imagem Docker, push para o Amazon ECR e
# criação/atualização do serviço no AWS App Runner.
#
# Pré-requisitos:
#   - AWS CLI configurado (aws configure)
#   - Docker instalado e em execução
#   - Permissões IAM para ECR e App Runner
#
# Uso:
#   chmod +x deploy.sh
#   ./deploy.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Variáveis de configuração — ajuste conforme seu ambiente
# ---------------------------------------------------------------------------
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO="${ECR_REPO:-passos-magicos-ml}"
APP_RUNNER_SERVICE="${APP_RUNNER_SERVICE:-passos-magicos-ml-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Obtém o ID da conta AWS automaticamente
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "============================================="
echo "  Deploy — Passos Mágicos ML API"
echo "============================================="
echo "Região AWS:       ${AWS_REGION}"
echo "Repositório ECR:  ${ECR_REPO}"
echo "Serviço App Runner: ${APP_RUNNER_SERVICE}"
echo "Tag da imagem:    ${IMAGE_TAG}"
echo "URI ECR:          ${ECR_URI}"
echo "============================================="

# ---------------------------------------------------------------------------
# 1. Criar repositório ECR (se não existir)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Verificando repositório ECR..."
if ! aws ecr describe-repositories \
    --repository-names "${ECR_REPO}" \
    --region "${AWS_REGION}" > /dev/null 2>&1; then
    echo "Repositório não encontrado. Criando '${ECR_REPO}'..."
    aws ecr create-repository \
        --repository-name "${ECR_REPO}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true
    echo "Repositório ECR criado com sucesso."
else
    echo "Repositório ECR já existe."
fi

# ---------------------------------------------------------------------------
# 2. Autenticar Docker no ECR
# ---------------------------------------------------------------------------
echo ""
echo ">>> Autenticando Docker no ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ---------------------------------------------------------------------------
# 3. Build da imagem Docker
# ---------------------------------------------------------------------------
echo ""
echo ">>> Construindo imagem Docker..."
docker build -t "${ECR_REPO}:${IMAGE_TAG}" .

# ---------------------------------------------------------------------------
# 4. Tag e push para o ECR
# ---------------------------------------------------------------------------
echo ""
echo ">>> Enviando imagem para o ECR..."
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:${IMAGE_TAG}"
echo "Imagem enviada: ${ECR_URI}:${IMAGE_TAG}"

# ---------------------------------------------------------------------------
# 5. Criar ou atualizar serviço App Runner
# ---------------------------------------------------------------------------
echo ""
echo ">>> Configurando serviço App Runner..."

# Verificar se o serviço já existe
SERVICE_ARN=$(aws apprunner list-services \
    --region "${AWS_REGION}" \
    --query "ServiceSummaryList[?ServiceName=='${APP_RUNNER_SERVICE}'].ServiceArn" \
    --output text 2>/dev/null || echo "")

if [ -z "${SERVICE_ARN}" ] || [ "${SERVICE_ARN}" = "None" ]; then
    echo "Criando novo serviço App Runner '${APP_RUNNER_SERVICE}'..."
    aws apprunner create-service \
        --region "${AWS_REGION}" \
        --service-name "${APP_RUNNER_SERVICE}" \
        --source-configuration "{
            \"ImageRepository\": {
                \"ImageIdentifier\": \"${ECR_URI}:${IMAGE_TAG}\",
                \"ImageRepositoryType\": \"ECR\",
                \"ImageConfiguration\": {
                    \"Port\": \"8000\"
                }
            },
            \"AutoDeploymentsEnabled\": true
        }" \
        --health-check-configuration "{
            \"Protocol\": \"HTTP\",
            \"Path\": \"/health\",
            \"Interval\": 10,
            \"Timeout\": 5,
            \"HealthyThreshold\": 1,
            \"UnhealthyThreshold\": 5
        }" \
        --instance-configuration "{
            \"Cpu\": \"1024\",
            \"Memory\": \"2048\"
        }"
    echo "Serviço App Runner criado com sucesso."
else
    echo "Atualizando serviço App Runner existente (${SERVICE_ARN})..."
    aws apprunner update-service \
        --region "${AWS_REGION}" \
        --service-arn "${SERVICE_ARN}" \
        --source-configuration "{
            \"ImageRepository\": {
                \"ImageIdentifier\": \"${ECR_URI}:${IMAGE_TAG}\",
                \"ImageRepositoryType\": \"ECR\",
                \"ImageConfiguration\": {
                    \"Port\": \"8000\"
                }
            },
            \"AutoDeploymentsEnabled\": true
        }" \
        --health-check-configuration "{
            \"Protocol\": \"HTTP\",
            \"Path\": \"/health\",
            \"Interval\": 10,
            \"Timeout\": 5,
            \"HealthyThreshold\": 1,
            \"UnhealthyThreshold\": 5
        }"
    echo "Serviço App Runner atualizado com sucesso."
fi

echo ""
echo "============================================="
echo "  Deploy concluído com sucesso!"
echo "============================================="
echo ""
echo "Para verificar o status do serviço:"
echo "  aws apprunner list-services --region ${AWS_REGION}"
echo ""
echo "Para obter a URL do serviço:"
echo "  aws apprunner describe-service --service-arn <SERVICE_ARN> --query 'Service.ServiceUrl'"
