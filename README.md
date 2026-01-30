# TextMining & LLMs - Lecture Notebooks

Laboratory notebooks for the **TextMining & LLMs** course at Alma Mater Studiorum Università di Bologna.

**Author**: Lorenzo Molfetta  
**Contacts**: lorenzo.molfetta@unibo.it

**Research Group**: UniboNLP@Cesena  
**Course Instructor**: Prof. Gianluca Moro

## Notebooks

| Notebook | Topics | Requirements |
|----------|--------|--------------|
| [Prompting & Fine-tuning](prompting_finetuning/) | Decoding strategies, prompt engineering, QLoRA, SFT, GRPO | [requirements.txt](prompting_finetuning/requirements.txt) |
| [RAG Chatbot](rag_chatbot/) | Vector stores, semantic retrieval, Milvus, ReAct agents | [requirements.txt](rag_chatbot/requirements.txt) |
| [LAM & Agents](lam_agents/) | Large Action Models, tool calling, MCP protocol | [requirements.txt](lam_agents/requirements.txt) |

Each notebook has its own `requirements.txt` with pinned versions.

**Available notebook IDs:**
- `lam_agents` - LAM & Agents
- `prompting_finetuning` - Prompting & Fine-tuning  
- `rag_chatbot` - RAG Chatbot

---

## Local Installation

Run notebooks on your own machine using Docker.

### Quick Start

```bash
# With GPU (default)
./start.sh lam_agents

# Without GPU (CPU-only, no NVIDIA required)
./start.sh rag_chatbot --cpu
```

The script builds the image, starts the container, and opens your browser at http://localhost:50000

### Commands

| Command | Description |
|---------|-------------|
| `./start.sh <notebook>` | Build & run with GPU |
| `./start.sh <notebook> --cpu` | Build & run CPU-only |
| `./start.sh stop` | Stop all containers |
| `./start.sh list` | List available notebooks |

### Requirements

- Docker
- NVIDIA Container Toolkit + GPU with CUDA 12.2+ (for GPU mode)

### Manual Python Environment

If not using Docker:

```bash
# Install dependencies for chosen notebook
cd lam_agents && pip install -r requirements.txt

# Start Jupyter
jupyter lab
```

Some notebooks require [Ollama](https://ollama.com) for local inference:

```bash
ollama serve &
ollama pull qwen3:1.7b
ollama pull llama3.2:3b
ollama pull bge-m3
```

---

## Remote Server

Connect to a GPU server via SSH and run notebooks there.

### 1. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
REMOTE_SERVER=137.204.107.40
REMOTE_USER=molfetta
SSH_PORT=22
HF_TOKEN=your_huggingface_token
```

### 2. Run a Notebook

```bash
# One command does everything: SSH → launch container → open tunnel → open browser
./run_notebook.sh connect lam_agents
```

Available notebooks: `lam_agents`, `prompting_finetuning`, `rag_chatbot`

### 3. Access Jupyter

Open http://localhost:50000 in your browser.

### Other Remote Commands

```bash
./run_notebook.sh status              # Check running containers
./run_notebook.sh stop                # Stop all remote containers
./run_notebook.sh tunnel              # Open tunnel only (container already running)
./run_notebook.sh launch <notebook>   # Launch container only (no tunnel)
```

### Custom Server

```bash
./run_notebook.sh connect lam_agents -s 192.168.1.100 -u john -p 22
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REMOTE_SERVER` | GPU server address | 137.204.107.40 |
| `REMOTE_USER` | SSH username | molfetta |
| `SSH_PORT` | SSH port | 22 |
| `LOCAL_PORT` | Local Jupyter port | 50000 |
| `REMOTE_PORT` | Remote Jupyter port | 50000 |
| `WORKSPACE_PATH` | Remote workspace | $HOME/workspace |
| `HF_TOKEN` | HuggingFace API token | - |
| `CUDA_VISIBLE_DEVICES` | GPU device to use | 0 |

### GPU Configuration

By default, containers use **GPU 0 only**. To change which GPU(s) are visible inside the container:

```bash
# Use GPU 1 instead
export CUDA_VISIBLE_DEVICES=1
./start.sh rag_chatbot

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
./run_notebook.sh connect lam_agents

# Or set in .env file
echo "CUDA_VISIBLE_DEVICES=1" >> .env
```

**Available values:**
- `0` - GPU 0 only (default)
- `1` - GPU 1 only
- `0,1` - GPUs 0 and 1
- `0,1,2` - GPUs 0, 1, and 2

### Customizing Connections

You can customize connection settings in multiple ways:

**1. Using `.env` file (Recommended)**
```bash
cp .env.example .env
# Edit .env with your values
```

**2. Using command-line arguments**
```bash
./run_notebook.sh connect lam_agents -s 192.168.1.100 -u username -p 22
```

**3. Using environment variables**
```bash
export LOCAL_PORT=50000
export REMOTE_PORT=50000
export SSH_PORT=22
./run_notebook.sh connect lam_agents
```

**Default Port Configuration:**
- **Local Port**: 50000 (port on your machine where Jupyter will be accessible)
- **Remote Port**: 50000 (port inside the Docker container running Jupyter)
- **SSH Port**: 22 (standard SSH port, may vary by server)

When connecting, the SSH tunnel maps `localhost:50000` on your machine to `localhost:50000` on the remote server, which forwards to port 50000 inside the Docker container.

**Port Customization Example:**
```bash
# Use different ports
export LOCAL_PORT=8888
export REMOTE_PORT=8888
./run_notebook.sh connect rag_chatbot
```

For manual setup instructions, see [utils/connect_jupyter_server.md](utils/connect_jupyter_server.md).

---

## Project Structure

```
├── start.sh                        # Build & run locally (main script)
├── run_notebook.sh                 # Remote server connection
├── Dockerfile                      # GPU container (nvidia/cuda:12.2.0)
├── Dockerfile.cpu                  # CPU container (python:3.11-slim)
├── .env.example                    # Environment configuration template
├── lam_agents/
│   ├── requirements.txt
│   └── TextMining&LLM_LAB_Agents.ipynb
├── prompting_finetuning/
│   ├── requirements.txt
│   └── TextMiningLLMs_Lecture_Prompting_Finetuning.ipynb
└── rag_chatbot/
    ├── requirements.txt
    └── TextMining_Lecture_RAG_Chatbot.ipynb
```

## License

Educational use. Contact UniboNLP for other uses.
