r"""
### Start it, once

The first run downloads the 6.7 GB GGUF through `huggingface_hub` and caches it on disk,
so a second run only starts the process. It is detached, because a server that dies with
the cell cannot be called by the next one. The wait is a poll of `/v1/models`, not a
sleep: the weights take about a minute to load on a T4, and guessing a duration is how
notebooks break on a slower machine. On timeout the last log lines are printed instead of
a stack trace, and every later cell checks `SERVER_READY` before calling the model.
"""

import subprocess
import sys
import time

from huggingface_hub import hf_hub_download

# 1. Download once (cached), then start the server detached from this cell.
if not server_ready():
    model_path = hf_hub_download(LLM_REPO, LLM_FILE, revision=LLM_REVISION)
    log = open("llama-server.log", "wb")
    subprocess.Popen(
        [sys.executable, "-m", "llama_cpp.server", *server_args(model_path)],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # survives the notebook process
    )
    for _ in range(300):
        if server_ready():
            break
        time.sleep(2)

# 2. One flag the next cells check, and the log tail when it never came up.
SERVER_READY = server_ready()
print("server ready:", SERVER_READY, "| log: llama-server.log")
if not SERVER_READY:
    tail = Path("llama-server.log").read_text(errors="replace").splitlines()[-20:]
    print("\n".join(tail))
    print(
        "\nThe server did not answer. Two likely causes:\n"
        "  - llama-cpp-python was installed without CUDA: rebuild with\n"
        "    CMAKE_ARGS='-DGGML_CUDA=on' pip install --no-binary llama-cpp-python llama-cpp-python\n"
        "  - the quant does not fit: switch LLM_FILE to a smaller GGUF and rerun this cell.\n"
        "Sections 15-17 keep running with stub answers, so the rest of the lecture still executes."
    )
