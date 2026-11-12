r"""
<a name="section-14"></a>

# 14. A reasoning model, served from Python

The retrieval side is done; this is the generator. A free Colab GPU has about 14 GB of
usable memory, so the model has to fit next to what the notebook already loaded:
`Gemma 4 12B` in Unsloth's QAT dynamic quant is 6.7 GB of weights, which leaves room for
an 8k context and the encoders.

The server is `llama-cpp-python`, installed with pip, so there is no release binary to
download and nothing to match against the image's glibc — the one failure mode that used
to break this section on older machines. `python -m llama_cpp.server` speaks the OpenAI
API, which is why the client in the next cell is four lines.

The GGUF was checked at `unsloth/gemma-4-12B-it-qat-GGUF` revision `980b060c`
(`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`, 6.72 GB). Quantization-aware training keeps it near
bfloat16 quality at four bits. If the runtime is tighter, `gemma-4-E4B-it-GGUF` is the
fallback.
"""

import shutil
import urllib.request
from pathlib import Path

# 1. The model, the endpoint, and the revision every later cell depends on.
LLM_MODEL = "gemma4-12b"
LLM_URL = "http://127.0.0.1:8080/v1"
LLM_REPO = "unsloth/gemma-4-12B-it-qat-GGUF"
LLM_FILE = "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
LLM_REVISION = "980b060c"
LLM_CTX = 8192


def server_args(model_path):
    """The flags that matter: all layers on the GPU, an 8k window, the model's own template."""
    return [
        "--model", str(model_path),
        "--alias", LLM_MODEL,
        "--host", "127.0.0.1",
        "--port", "8080",
        "--n_ctx", str(LLM_CTX),
        "--n_gpu_layers", "-1" if shutil.which("nvidia-smi") else "0",
        "--chat_format", "gemma",
    ]


def server_ready(timeout=2):
    """The OpenAI route answers only once the weights are loaded."""
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=timeout)
        return True
    except OSError:
        return False


print("model", LLM_MODEL, "| endpoint", LLM_URL, "| installed:", shutil.which("python") is not None)
