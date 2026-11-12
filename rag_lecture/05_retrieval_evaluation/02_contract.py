r"""
### The one function every retriever goes through

`evaluate_run` is the contract of the whole lecture, so it is worth reading before any score appears. It refuses three things, because each of them would silently produce a better number than the retriever earned: a ranking that does not cover exactly the test questions, a ranking shorter than `MAX_K` or with duplicate or unknown document IDs, and a non-finite score. The first helper below validates one question's hits and imposes one total order, score descending and document ID ascending, so ties are resolved the same way in every section rather than by whatever the library happened to return. The two objects this cell
opens, `all_rankings` and `evaluation_rows`, are the ledgers every later section writes to.
"""

# Both helpers below are called by every section, so they live with the ledger.
import math

all_rankings = {}
evaluation_rows = []


def canonicalize(qid, hits):
    """Validate one question's hits and return the first MAX_K in the lecture's order."""
    ids = [hit["corpus_id"] for hit in hits]
    if len(ids) < MAX_K or len(ids) != len(set(ids)) or not set(ids) <= set(corpus):
        raise ValueError(f"{qid}: need {MAX_K} distinct corpus IDs, got {len(ids)}")
    if any(not math.isfinite(float(hit["score"])) for hit in hits):
        raise ValueError(f"{qid}: non-finite score")
    ordered = sorted(hits, key=lambda hit: (-float(hit["score"]), hit["corpus_id"]))
    return [{"corpus_id": hit["corpus_id"], "score": float(hit["score"])} for hit in ordered][:MAX_K]


print("canonicalize ready: validity checks first, then one total order.")
