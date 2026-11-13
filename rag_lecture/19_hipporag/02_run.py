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
import shutil
import subprocess
import sys

# HippoRAG extracts triples with an LLM, so without a local server there is nothing to build.
hippo_result = None
if not SERVER_READY:
    print("skipped: HippoRAG's OpenIE pass needs the local model from section 14.")
else:
    # Prefer the isolated uv project; fall back to this interpreter when uv is absent.
    runner = ["uv", "run", "--project", "hippo_backend", "--frozen"] if shutil.which("uv") else [sys.executable]
    subprocess.run(
        [*runner, "hippo_backend/run.py",
         "--input", str(hippo_input), "--output", str(hippo_output)],
        check=True,
    )

    # Refuse a stale result whose input hash does not match the current file.
    hippo_result = json.loads(hippo_output.read_text(encoding="utf-8"))
    if hippo_result["input_sha256"] != hashlib.sha256(hippo_input.read_bytes()).hexdigest():
        raise ValueError("Backend output does not match the current input")

    # The graph statistics this index produced.
    display({"graph_info": hippo_result["graph_info"],
             "index_fingerprint": hippo_result["index_fingerprint"]})
