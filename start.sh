#!/bin/bash
# =============================================================================
# start.sh - Build and run Jupyter Lab for TextMining & LLMs notebooks
# =============================================================================
# UniboNLP@Cesena - Alma Mater Studiorum Università di Bologna
# =============================================================================

set -e

# Load .env if exists
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    source "${SCRIPT_DIR}/.env"
fi

# Configuration
IMAGE_NAME="unibonlp/textmining-llm-notebooks"
CONTAINER_NAME="textmining-llm-lab"
JUPYTER_PORT="${JUPYTER_PORT:-50000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Available notebooks
NOTEBOOKS=("lam_agents" "prompting_finetuning" "rag_chatbot")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo ""
    echo -e "${BOLD}Usage:${NC} $0 <notebook> [--cpu]"
    echo ""
    echo -e "${BOLD}Notebooks:${NC}"
    echo "  lam_agents            LAM & Agents (LlamaIndex, MCP)"
    echo "  prompting_finetuning  Prompting & Fine-tuning (SFT, GRPO)"
    echo "  rag_chatbot           RAG Chatbot (Milvus, LangChain)"
    echo ""
    echo -e "${BOLD}Options:${NC}"
    echo "  --cpu    Build and run without GPU (uses python:3.11-slim)"
    echo "           Default: GPU mode (uses nvidia/cuda:12.2.0)"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  $0 lam_agents          # Build & run with GPU"
    echo "  $0 rag_chatbot --cpu   # Build & run CPU-only"
    echo ""
    echo -e "${BOLD}Other commands:${NC}"
    echo "  $0 stop                # Stop running container"
    echo "  $0 list                # List available notebooks"
    echo ""
    exit 1
}

validate_notebook() {
    local notebook="$1"
    for n in "${NOTEBOOKS[@]}"; do
        if [[ "$n" == "$notebook" ]]; then
            return 0
        fi
    done
    print_error "Unknown notebook: $notebook"
    echo "Available notebooks: ${NOTEBOOKS[*]}"
    exit 1
}

stop_containers() {
    print_info "Stopping containers..."
    for notebook in "${NOTEBOOKS[@]}"; do
        docker stop "${CONTAINER_NAME}-${notebook}" 2>/dev/null || true
        docker rm "${CONTAINER_NAME}-${notebook}" 2>/dev/null || true
    done
    print_info "Done."
    exit 0
}

list_notebooks() {
    echo ""
    echo "Available notebooks:"
    for n in "${NOTEBOOKS[@]}"; do
        echo "  - $n"
    done
    echo ""
    exit 0
}

# Parse arguments
NOTEBOOK=""
USE_CPU=false

for arg in "$@"; do
    case "$arg" in
        --cpu)
            USE_CPU=true
            ;;
        stop)
            stop_containers
            ;;
        list)
            list_notebooks
            ;;
        -h|--help)
            usage
            ;;
        *)
            NOTEBOOK="$arg"
            ;;
    esac
done

# Validate
if [[ -z "$NOTEBOOK" ]]; then
    usage
fi
validate_notebook "$NOTEBOOK"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed."
    exit 1
fi

# Determine image tag and Dockerfile
if [[ "$USE_CPU" == true ]]; then
    IMAGE_TAG="${IMAGE_NAME}:${NOTEBOOK}-cpu"
    DOCKERFILE="${SCRIPT_DIR}/Dockerfile.cpu"
    GPU_FLAG=""
    print_info "Mode: CPU-only"
else
    IMAGE_TAG="${IMAGE_NAME}:${NOTEBOOK}"
    DOCKERFILE="${SCRIPT_DIR}/Dockerfile"
    GPU_FLAG="--gpus '\"device=${CUDA_VISIBLE_DEVICES}\"'"
    print_info "Mode: GPU (device=${CUDA_VISIBLE_DEVICES})"
fi

CONTAINER="${CONTAINER_NAME}-${NOTEBOOK}"

# Stop existing container
docker rm -f "${CONTAINER}" 2>/dev/null || true

# Build
echo ""
print_info "Building image: ${IMAGE_TAG}"
print_info "This may take a few minutes on first run..."
echo ""

TMP_DOCKERFILE=$(mktemp)
sed "s|COPY requirements.txt|COPY ${NOTEBOOK}/requirements.txt|" "${DOCKERFILE}" > "$TMP_DOCKERFILE"

docker build -t "${IMAGE_TAG}" -f "$TMP_DOCKERFILE" "${SCRIPT_DIR}"
rm "$TMP_DOCKERFILE"

print_info "Build complete!"
echo ""

# Run
print_info "Starting container: ${CONTAINER}"

if [[ "$USE_CPU" == true ]]; then
    docker run -d \
        --name "${CONTAINER}" \
        -p "${JUPYTER_PORT}:8888" \
        -v "${SCRIPT_DIR}/${NOTEBOOK}:/workspace/${NOTEBOOK}" \
        -w "/workspace/${NOTEBOOK}" \
        "${IMAGE_TAG}"
else
    docker run -d \
        --name "${CONTAINER}" \
        --gpus "device=${CUDA_VISIBLE_DEVICES}" \
        -p "${JUPYTER_PORT}:8888" \
        -v "${SCRIPT_DIR}/${NOTEBOOK}:/workspace/${NOTEBOOK}" \
        -w "/workspace/${NOTEBOOK}" \
        "${IMAGE_TAG}" \
        jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser \
            --NotebookApp.token='' --NotebookApp.password=''
fi

# Wait for Jupyter to start
sleep 2

# Output
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}Jupyter Lab is running!${NC}"
echo ""
echo -e "  ${CYAN}➜  http://localhost:${JUPYTER_PORT}${NC}"
echo ""
echo -e "  Notebook: ${BOLD}${NOTEBOOK}${NC}"
echo -e "  Container: ${CONTAINER}"
echo ""
echo -e "  Stop with: ${YELLOW}./start.sh stop${NC}"
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
