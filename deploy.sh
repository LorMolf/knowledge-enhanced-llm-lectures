#!/bin/bash
# =============================================================================
# deploy.sh - Deploy Jupyter Lab environment for TextMining & LLMs lectures
# =============================================================================
# UniboNLP@Cesena - Alma Mater Studiorum Università di Bologna
# =============================================================================

set -e

# Configuration
IMAGE_NAME="unibonlp/textmining-llm-notebooks"
CONTAINER_NAME="textmining-llm-lab"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Available notebooks
NOTEBOOKS=("lam_agents" "prompting_finetuning" "rag_chatbot")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS] [NOTEBOOK]"
    echo ""
    echo "Commands:"
    echo "  build [--cpu] <notebook>  Build Docker image for a specific notebook"
    echo "  build-all [--cpu]         Build Docker images for all notebooks"
    echo "  run <notebook>            Run container with GPU support"
    echo "  run-cpu <notebook>        Run container without GPU (CPU only)"
    echo "  stop                      Stop the running container"
    echo "  logs <notebook>           Show container logs"
    echo "  shell <notebook>          Open a shell in the running container"
    echo "  clean                     Remove container and image"
    echo "  list                      List available notebooks"
    echo ""
    echo "Build Options:"
    echo "  --cpu                 Build with CPU-only base image (python:3.11-slim)"
    echo "                        Default: GPU image (nvidia/cuda:12.2.0-devel-ubuntu22.04)"
    echo ""
    echo "Notebooks:"
    echo "  lam_agents            LAM & Agents (LlamaIndex, MCP)"
    echo "  prompting_finetuning  Prompting & Fine-tuning (SFT, GRPO)"
    echo "  rag_chatbot           RAG Chatbot (Milvus, LangChain)"
    echo ""
    echo "Environment variables:"
    echo "  JUPYTER_PORT    Port for Jupyter Lab (default: 8888)"
    echo "  CUDA_VISIBLE_DEVICES    GPU device(s) to use"
    echo ""
    exit 1
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
}

check_nvidia_docker() {
    if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        print_warn "NVIDIA Container Toolkit not detected. GPU support may not work."
        print_warn "Install it from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        return 1
    fi
    return 0
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

list_notebooks() {
    echo "Available notebooks:"
    for n in "${NOTEBOOKS[@]}"; do
        echo "  - $n"
    done
}

build() {
    local notebook=""
    local use_cpu=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --cpu)
                use_cpu=true
                shift
                ;;
            *)
                notebook="$1"
                shift
                ;;
        esac
    done
    
    if [[ -z "$notebook" ]]; then
        print_error "Please specify a notebook: ./deploy.sh build [--cpu] <notebook>"
        list_notebooks
        exit 1
    fi
    validate_notebook "$notebook"
    
    local image="${IMAGE_NAME}:${notebook}"
    local dockerfile="${WORKSPACE_DIR}/Dockerfile"
    
    if [[ "$use_cpu" == true ]]; then
        dockerfile="${WORKSPACE_DIR}/Dockerfile.cpu"
        image="${IMAGE_NAME}:${notebook}-cpu"
        print_info "Building CPU-only Docker image: ${image}"
    else
        print_info "Building GPU Docker image: ${image}"
    fi
    
    # Create temporary Dockerfile with notebook-specific requirements
    local tmp_dockerfile=$(mktemp)
    sed "s|COPY requirements.txt|COPY ${notebook}/requirements.txt|" "${dockerfile}" > "$tmp_dockerfile"
    
    docker build -t "${image}" -f "$tmp_dockerfile" "${WORKSPACE_DIR}"
    rm "$tmp_dockerfile"
    
    print_info "Build complete: ${image}"
}

build_all() {
    local use_cpu=false
    
    if [[ "$1" == "--cpu" ]]; then
        use_cpu=true
    fi
    
    for notebook in "${NOTEBOOKS[@]}"; do
        if [[ "$use_cpu" == true ]]; then
            build --cpu "$notebook"
        else
            build "$notebook"
        fi
    done
}

run_gpu() {
    local notebook="$1"
    if [[ -z "$notebook" ]]; then
        print_error "Please specify a notebook: ./deploy.sh run <notebook>"
        list_notebooks
        exit 1
    fi
    validate_notebook "$notebook"
    
    check_nvidia_docker || print_warn "Continuing without GPU verification..."
    
    local image="${IMAGE_NAME}:${notebook}"
    local container="${CONTAINER_NAME}-${notebook}"
    
    # Stop existing container if running
    docker rm -f "${container}" 2>/dev/null || true
    
    GPU_FLAG=""
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        GPU_FLAG="--gpus device=${CUDA_VISIBLE_DEVICES}"
    else
        GPU_FLAG="--gpus all"
    fi
    
    print_info "Starting container with GPU support..."
    docker run -d \
        --name "${container}" \
        ${GPU_FLAG} \
        -p "${JUPYTER_PORT}:8888" \
        -m 30g \
        -v "${WORKSPACE_DIR}:/workspace" \
        "${image}" \
        jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser \
            --NotebookApp.token='' --NotebookApp.password=''
    
    print_info "Container started: ${container}"
    print_info "Jupyter Lab available at: http://localhost:${JUPYTER_PORT}"
}

run_cpu() {
    local notebook="$1"
    if [[ -z "$notebook" ]]; then
        print_error "Please specify a notebook: ./deploy.sh run-cpu <notebook>"
        list_notebooks
        exit 1
    fi
    validate_notebook "$notebook"
    
    local image="${IMAGE_NAME}:${notebook}"
    local container="${CONTAINER_NAME}-${notebook}"
    
    # Stop existing container if running
    docker rm -f "${container}" 2>/dev/null || true
    
    print_info "Starting container (CPU only)..."
    docker run -d \
        --name "${container}" \
        -p "${JUPYTER_PORT}:8888" \
        -m 30g \
        -v "${WORKSPACE_DIR}:/workspace" \
        "${image}" \
        jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser \
            --NotebookApp.token='' --NotebookApp.password=''
    
    print_info "Container started: ${container}"
    print_info "Jupyter Lab available at: http://localhost:${JUPYTER_PORT}"
}

stop() {
    print_info "Stopping containers..."
    for notebook in "${NOTEBOOKS[@]}"; do
        docker stop "${CONTAINER_NAME}-${notebook}" 2>/dev/null || true
        docker rm "${CONTAINER_NAME}-${notebook}" 2>/dev/null || true
    done
    print_info "Containers stopped."
}

logs() {
    local notebook="$1"
    if [[ -z "$notebook" ]]; then
        print_error "Please specify a notebook: ./deploy.sh logs <notebook>"
        exit 1
    fi
    docker logs -f "${CONTAINER_NAME}-${notebook}"
}

shell() {
    local notebook="$1"
    if [[ -z "$notebook" ]]; then
        print_error "Please specify a notebook: ./deploy.sh shell <notebook>"
        exit 1
    fi
    docker exec -it "${CONTAINER_NAME}-${notebook}" bash
}

clean() {
    print_info "Cleaning up..."
    for notebook in "${NOTEBOOKS[@]}"; do
        docker rm -f "${CONTAINER_NAME}-${notebook}" 2>/dev/null || true
        docker rmi "${IMAGE_NAME}:${notebook}" 2>/dev/null || true
    done
    print_info "Cleanup complete."
}

# Main
check_docker

case "${1:-}" in
    build)
        shift
        build "$@"
        ;;
    build-all)
        shift
        build_all "$@"
        ;;
    run)
        run_gpu "$2"
        ;;
    run-cpu)
        run_cpu "$2"
        ;;
    stop)
        stop
        ;;
    logs)
        logs "$2"
        ;;
    shell)
        shell "$2"
        ;;
    clean)
        clean
        ;;
    list)
        list_notebooks
        ;;
    *)
        usage
        ;;
esac
