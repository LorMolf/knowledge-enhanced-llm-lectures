#!/bin/bash
# =============================================================================
# run_notebook.sh - Connect to remote server and launch Jupyter Lab
# =============================================================================
# UniboNLP@Cesena - Alma Mater Studiorum Università di Bologna
# =============================================================================

set -e

# Load .env if exists
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    source "${SCRIPT_DIR}/.env"
fi

# Default configuration (can be overridden by .env)
DEFAULT_SERVER="${REMOTE_SERVER:-137.204.107.40}"
DEFAULT_USER="${REMOTE_USER:-molfetta}"
DEFAULT_PORT="${SSH_PORT:-22}"
LOCAL_PORT="${LOCAL_PORT:-50000}"
REMOTE_PORT="${REMOTE_PORT:-50000}"
WORKSPACE_PATH="${WORKSPACE_PATH:-\$HOME/workspace}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_cmd() { echo -e "${CYAN}[CMD]${NC} $1"; }

usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  tunnel              Open SSH tunnel to remote Jupyter server"
    echo "  launch <notebook>   SSH into server and launch Jupyter with notebook"
    echo "  connect <notebook>  Launch on server + open tunnel (full setup)"
    echo "  stop                Stop remote Jupyter container"
    echo "  status              Check if remote container is running"
    echo ""
    echo "Notebooks:"
    echo "  lam_agents            LAM & Agents"
    echo "  prompting_finetuning  Prompting & Fine-tuning"
    echo "  rag_chatbot           RAG Chatbot"
    echo ""
    echo "Options:"
    echo "  -s, --server HOST   Remote server (default: ${DEFAULT_SERVER})"
    echo "  -u, --user USER     SSH username (default: ${DEFAULT_USER})"
    echo "  -p, --port PORT     SSH port (default: ${DEFAULT_PORT})"
    echo ""
    echo "Environment variables:"
    echo "  LOCAL_PORT            Local port for tunnel (default: 50000)"
    echo "  REMOTE_PORT           Remote Jupyter port (default: 50000)"
    echo "  WORKSPACE_PATH        Remote workspace path (default: \$HOME/workspace)"
    echo "  CUDA_VISIBLE_DEVICES  GPU device to use (default: 0)"
    echo ""
    echo "Examples:"
    echo "  $0 connect lam_agents"
    echo "  $0 tunnel -s 192.168.1.100 -u john"
    echo "  $0 launch rag_chatbot --server myserver.edu"
    echo ""
    exit 1
}

# Parse arguments
SERVER="${DEFAULT_SERVER}"
USER="${DEFAULT_USER}"
SSH_PORT="${DEFAULT_PORT}"
NOTEBOOK=""
COMMAND=""

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            tunnel|launch|connect|stop|status)
                COMMAND="$1"
                shift
                ;;
            lam_agents|prompting_finetuning|rag_chatbot)
                NOTEBOOK="$1"
                shift
                ;;
            -s|--server)
                SERVER="$2"
                shift 2
                ;;
            -u|--user)
                USER="$2"
                shift 2
                ;;
            -p|--port)
                SSH_PORT="$2"
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                ;;
        esac
    done
}

check_ssh() {
    if ! command -v ssh &> /dev/null; then
        print_error "SSH is not installed."
        exit 1
    fi
}

kill_existing_tunnel() {
    # Kill any existing tunnel on the local port
    local pid=$(lsof -ti:${LOCAL_PORT} 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        print_warn "Killing existing process on port ${LOCAL_PORT} (PID: $pid)"
        kill -9 $pid 2>/dev/null || true
        sleep 1
    fi
}

open_tunnel() {
    kill_existing_tunnel
    
    print_info "Opening SSH tunnel: localhost:${LOCAL_PORT} -> ${SERVER}:${REMOTE_PORT}"
    print_cmd "ssh -N -f -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} -p ${SSH_PORT} ${USER}@${SERVER}"
    
    ssh -N -f -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} -p ${SSH_PORT} ${USER}@${SERVER}
    
    if [[ $? -eq 0 ]]; then
        print_info "Tunnel established!"
        print_info "Access Jupyter at: http://localhost:${LOCAL_PORT}"
    else
        print_error "Failed to establish tunnel"
        exit 1
    fi
}

launch_jupyter() {
    local notebook="$1"
    
    if [[ -z "$notebook" ]]; then
        print_error "Please specify a notebook: ./remote.sh launch <notebook>"
        echo "Available: lam_agents, prompting_finetuning, rag_chatbot"
        exit 1
    fi
    
    local image="unibonlp/textmining-llm-notebooks:${notebook}"
    local container="textmining-llm-lab-${notebook}"
    
    print_info "Launching Jupyter on ${SERVER} for notebook: ${notebook} (GPU: ${CUDA_VISIBLE_DEVICES})"
    
    # SSH command to run on remote
    local remote_cmd="
        docker rm -f ${container} 2>/dev/null || true
        docker run -d \\
            --name ${container} \\
            --gpus '\"device=${CUDA_VISIBLE_DEVICES}\"' \\
            -p ${REMOTE_PORT}:8888 \\
            -m 30g \\
            -v ${WORKSPACE_PATH}:/workspace \\
            ${image} \\
            jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser \\
                --NotebookApp.token='' --NotebookApp.password=''
        echo 'Container started: ${container} (GPU: ${CUDA_VISIBLE_DEVICES})'
    "
    
    print_cmd "ssh -p ${SSH_PORT} ${USER}@${SERVER} '...'"
    ssh -p ${SSH_PORT} ${USER}@${SERVER} "${remote_cmd}"
    
    if [[ $? -eq 0 ]]; then
        print_info "Jupyter container launched on remote server"
    else
        print_error "Failed to launch container"
        exit 1
    fi
}

full_connect() {
    local notebook="$1"
    
    if [[ -z "$notebook" ]]; then
        print_error "Please specify a notebook: ./remote.sh connect <notebook>"
        echo "Available: lam_agents, prompting_finetuning, rag_chatbot"
        exit 1
    fi
    
    print_info "=== Full Remote Setup ==="
    echo ""
    
    # Step 1: Launch on remote
    launch_jupyter "$notebook"
    
    echo ""
    sleep 2
    
    # Step 2: Open tunnel
    open_tunnel
    
    echo ""
    print_info "=== Setup Complete ==="
    print_info "Jupyter Lab: http://localhost:${LOCAL_PORT}"
    print_info "Notebook: ${notebook}"
    echo ""
    
    # Open browser (macOS)
    if [[ "$(uname)" == "Darwin" ]]; then
        print_info "Opening browser..."
        sleep 2
        open "http://localhost:${LOCAL_PORT}"
    fi
}

stop_remote() {
    print_info "Stopping remote containers..."
    
    local remote_cmd="
        docker stop textmining-llm-lab-lam_agents 2>/dev/null || true
        docker stop textmining-llm-lab-prompting_finetuning 2>/dev/null || true
        docker stop textmining-llm-lab-rag_chatbot 2>/dev/null || true
        docker rm textmining-llm-lab-lam_agents 2>/dev/null || true
        docker rm textmining-llm-lab-prompting_finetuning 2>/dev/null || true
        docker rm textmining-llm-lab-rag_chatbot 2>/dev/null || true
        echo 'Containers stopped'
    "
    
    ssh -p ${SSH_PORT} ${USER}@${SERVER} "${remote_cmd}"
    
    # Kill local tunnel
    kill_existing_tunnel
    print_info "Done"
}

check_status() {
    print_info "Checking remote container status..."
    
    ssh -p ${SSH_PORT} ${USER}@${SERVER} "docker ps --filter 'name=textmining-llm-lab' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
}

# Main
check_ssh
parse_args "$@"

case "${COMMAND}" in
    tunnel)
        open_tunnel
        ;;
    launch)
        launch_jupyter "$NOTEBOOK"
        ;;
    connect)
        full_connect "$NOTEBOOK"
        ;;
    stop)
        stop_remote
        ;;
    status)
        check_status
        ;;
    *)
        usage
        ;;
esac
