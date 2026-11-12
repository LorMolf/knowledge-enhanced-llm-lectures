r"""
<a name="section-14"></a>

# 14. A reasoning model through llama-server

The retrieval side is done; this is the generator. A free Colab GPU has about 14 GB
of usable memory, so the model must fit next to everything the notebook has already
loaded: `Gemma 4 12B` in Unsloth's QAT dynamic quant is 6.7 GB of weights, which
leaves room for an 8k context and the encoders. It is also a *reasoning* model with a
switch: the same weights either think before answering or answer directly, and
llama.cpp exposes that switch both at launch and per request.

This cell pins the llama.cpp release, the model and the endpoint. The flags worth
reading: `--jinja` uses the model's own chat template, `--reasoning auto` lets the
template decide whether thinking is on by default, and `--reasoning-format auto`
parses the thought channel into a separate `reasoning_content` field instead of
leaving it in the answer text.

The GGUF was checked at `unsloth/gemma-4-12B-it-qat-GGUF` revision `980b060c`; the flags
below pin the quant name only, so re-check that revision if the numbers move
(`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`, 6.72 GB). Quantization-aware training keeps it
near bfloat16 quality at four bits, and the dynamic recipe spends its bits on the
layers that hurt most. `-hf` follows the repository head, so re-check that hash if the numbers move;
the flag pins the quant name, not the revision, so a later upload can change these numbers. A 14 GB T4 holds 6.7 GB of weights plus about 1.4 GB of KV cache at 8k
context; `gemma-4-E4B-it-GGUF` is the fallback when a runtime is tighter.
"""

import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

# 1. Pin the exact release and the endpoint every later cell calls.
LLAMA_VERSION = "b11960"
LLAMA_ASSET = f"llama-{LLAMA_VERSION}-bin-ubuntu-cuda-12.8-x64.tar.gz"
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/" + f"{LLAMA_VERSION}/{LLAMA_ASSET}"
LLAMA_DIR = Path(f"llama-{LLAMA_VERSION}")

LLM_URL = "http://127.0.0.1:8080/v1"
LLM_MODEL = "gemma4-12b"

# 2. 6.7 GB of weights, 8k context, all layers on the T4.
LLAMA_FLAGS = [
    "-hf", "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL",
    "--alias", LLM_MODEL,
    "--host", "127.0.0.1",
    "--port", "8080",
    "--jinja",                # the model's own template, required for tool calls
    "--reasoning", "auto",    # thinking on/off is decided by the template
    "--reasoning-format", "auto",  # the thought channel becomes reasoning_content
    "-c", "8192",
    "--n-gpu-layers", "99" if shutil.which("nvidia-smi") else "0",
]


def server_ready(timeout=2):
    """The /props endpoint answers only once the weights are loaded."""
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/props", timeout=timeout)
        return True
    except OSError:
        return False


print("pinned", LLAMA_VERSION, "| model", LLM_MODEL, "| endpoint", LLM_URL)
