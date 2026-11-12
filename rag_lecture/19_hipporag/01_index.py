r"""
<a name="section-19"></a>

# 19. HippoRAG 2: retrieval over connected evidence

HippoRAG extracts relations from papers, links question-related facts, and uses
personalized PageRank with passage evidence. The section 18 GLiNER graph is
not its index. ![Graph retrieval](assets/graph-retrieval.png)

HippoRAG's pinned dependencies conflict with the main notebook environment.
The next cell invokes the **separately installed** `hippo_backend` uv
project and exchanges explicit JSON files. It sends every common-corpus paper
to the section 14 loopback server for OpenIE, then indexes and retrieves; this
can take substantial time and storage. It runs only when you execute that cell.

The backend uses the same BGE checkpoint as section 07. Unlike that section's
prompted query baseline, HippoRAG's Transformers provider ignores query
instruction arguments and encodes queries without the BGE prefix. In the next
evaluation we also use its native dense passage retrieval as a no-prefix control.
Preprocessing, sequence lengths, library versions and graph settings can still
differ; no difference can be attributed to the graph alone.

Set `HIPPO_LLM_INSTANCE` to a new descriptive value whenever the server's
weights, quantization or template change: the cache cannot inspect the server's
actual model identity. The corpus and declared model/endpoint/instance identity
form the index fingerprint. Source: [pinned HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG/tree/d5c8329422e0a0b834a15874545cb6a74b4f9b26).

**Question:** what does the JSON handoff file contain, and how many papers and
questions does it carry?
"""

import json
from pathlib import Path

# 1. identity + paths: the instance tag is part of the index fingerprint
HIPPO_LLM_INSTANCE = "gemma-4-12B-it-QAT-UD-Q4_K_XL"
hippo_io = Path("outputs/hipporag-lecture-io")
hippo_io.mkdir(parents=True, exist_ok=True)
hippo_input = hippo_io / "input.json"
hippo_output = hippo_io / "output.json"

# 2. build the handoff payload the backend will read
hippo_task = {
    "corpus": [{"id": doc_id, "text": corpus[doc_id]} for doc_id in corpus_ids],
    "queries": [{"id": qid, "text": queries[qid]} for qid in query_ids],
    "k": MAX_K,
    "llm_base_url": LLM_URL,
    "llm_model": LLM_MODEL,
    "llm_instance": HIPPO_LLM_INSTANCE,
    "embedding_model": BGE_MODEL_ID,
}

# 3. write it to disk so the separate uv project can pick it up
hippo_input.write_text(json.dumps(hippo_task, ensure_ascii=False, indent=2), encoding="utf-8")

# 4. show what was handed off
display(
    {
        "input": str(hippo_input),
        "papers": len(hippo_task["corpus"]),
        "questions": len(hippo_task["queries"]),
        "model_instance": HIPPO_LLM_INSTANCE,
    }
)
