r"""
### Start it, once

The first run downloads the 6.7 GB GGUF through `huggingface_hub` and caches it, so a second
run only starts the process. Both steps are guarded: a server that cannot import will not
start, so the module is checked before the download, and the wait polls `/v1/models` while
watching the process, because a missing file or a bad flag kills it in a second and a blind
ten-minute sleep would hide the reason. On failure the log tail is printed and every later
cell checks `SERVER_READY` before calling the model.
"""

import importlib.util
import subprocess
import sys
import time

from huggingface_hub import hf_hub_download

# 1. Check the import first: otherwise the download happens for a server that cannot start.
LLAMA_INSTALLED = importlib.util.find_spec("llama_cpp") is not None
SERVER_READY = server_ready()

# 2. Download once (cached), start detached, then poll the route and the process together.
if not SERVER_READY and LLAMA_INSTALLED:
    model_path = hf_hub_download(LLM_REPO, LLM_FILE, revision=LLM_REVISION)
    log = open("llama-server.log", "wb")
    server = subprocess.Popen(
        [sys.executable, "-m", "llama_cpp.server", *server_args(model_path)],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # survives the notebook process
    )
    for _ in range(300):
        if server_ready() or server.poll() is not None:
            break
        time.sleep(2)
    SERVER_READY = server_ready()

print("llama-cpp-python installed:", LLAMA_INSTALLED, "| server ready:", SERVER_READY)
if not LLAMA_INSTALLED:
    print("run the install cell, restart the runtime, then rerun this cell; "
          "sections 15-17 fall back to stub answers until then.")
elif not SERVER_READY:
    print("\n".join(Path("llama-server.log").read_text(errors="replace").splitlines()[-20:]))
    print("\nThe server did not answer: switch LLM_FILE to a smaller GGUF, or rebuild with "
          "CMAKE_ARGS='-DGGML_CUDA=on' pip install --no-binary llama-cpp-python llama-cpp-python")
