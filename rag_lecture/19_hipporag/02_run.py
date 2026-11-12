r"""
### Build the isolated index

Run the backend on the JSON just written, with the loopback server already
started. It uses native OpenIE and retrieves the full common corpus before
breaking ties by original ID and cutting to k. Asking for full return depth
changes the returned slice, not native graph or dense scores; this is suitable
for the small lecture corpus, not a scalable production strategy. The input
hash protects us from accidentally reading a stale result file. What graph
statistics does this index produce?
"""

import hashlib
import subprocess

# 1. run the backend on the JSON we just wrote; --frozen pins the lockfile
subprocess.run(
    [
        "uv",
        "run",
        "--project",
        "hippo_backend",
        "--frozen",
        "hippo_backend/run.py",
        "--input",
        str(hippo_input),
        "--output",
        str(hippo_output),
    ],
    check=True,
)

# 2. load the result the backend wrote to disk
hippo_result = json.loads(hippo_output.read_text(encoding="utf-8"))

# 3. guard: refuse a stale result whose input hash does not match the current file
if hippo_result["input_sha256"] != hashlib.sha256(hippo_input.read_bytes()).hexdigest():
    raise ValueError("Backend output does not match the current input")

# 4. show the graph statistics and fingerprint this index produced
display(
    {
        "graph_info": hippo_result["graph_info"],
        "index_fingerprint": hippo_result["index_fingerprint"],
    }
)
