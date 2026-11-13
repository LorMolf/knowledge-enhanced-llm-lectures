r"""
### Ninety-six answers, written once

Three conditions per question, identical prompt and decoding, only the evidence list
changes. That is 96 short calls, so the frame is cached to parquet and read back on the
next run; `RUN_LLM = False` reuses the cache. `answer_row` builds one row so the loop
below stays five lines: question, evidence, answer, and the three label-free counts.

"""

import json
from pathlib import Path

# one switch and one cache path for the whole section
RUN_LLM = True  # False reuses the cached answers instead of paying 96 calls
RAG_CACHE = Path("outputs/rag_frame.parquet")
CONDITIONS = ("no context", "BGE top-3", "labelled papers")


def answer_row(qid, condition, evidence):
    """One (question, condition) row: the answer plus what it named, without labels."""
    question = f"Question: {queries[qid]}\nEvidence: {json.dumps(evidence)}"
    answer_text = chat([rag_system, {"role": "user", "content": question}]).get("content") or ""
    supplied = {entry["id"] for entry in evidence}
    cited = {doc_id for doc_id in supplied if doc_id in answer_text}
    return {
        "qid": qid,
        "condition": condition,
        "answer": answer_text,
        "cited supplied papers": len(cited),
        "cited labelled papers": len(cited & relevant_docs[qid]),
        "said insufficient": "insufficient" in answer_text.casefold(),
    }


print("answer_row ready | conditions:", CONDITIONS)
