r"""
### Rank the common test questions

The backend returns original paper IDs, not positions or graph hashes, and a
mode for every question. `dense_fallback` means the native retriever found no
facts after reranking and used dense passage retrieval; `graph_seeded` reports
selected facts, not proof that graph reasoning improved the answer. Compare the
native no-prefix dense control and HippoRAG with the earlier prompted BGE run.
All three use the shared evaluator. The first query's predicted seed triples
are linked back to source paper IDs and texts using HippoRAG's native OpenIE
provenance. Can you find the predicted relation in the source? Linked papers do
not by themselves establish that a question requires multi-hop reasoning.
"""

# 1. validate the backend response: one row per query, no duplicates, known modes
if hippo_result is None:
    hippo_records = []
    print("skipped: no HippoRAG index, the retrieval comparison below keeps its other rows.")
else:
    hippo_records = hippo_result["results"]
if (
    len(hippo_records) != len(query_ids)
    or len({row["id"] for row in hippo_records}) != len(hippo_records)
    or {row["id"] for row in hippo_records} != set(query_ids)
):
    raise ValueError("Backend returned missing, duplicate or unexpected query IDs")
if any(row["mode"] not in {"dense_fallback", "graph_seeded"} for row in hippo_records):
    raise ValueError("Backend did not identify every retrieval mode")

# 2. split the response into per-query rankings and modes
hippo_rankings = {row["id"]: row["hipporag"] for row in hippo_records}
hippo_dense_rankings = {row["id"]: row["dense"] for row in hippo_records}
hippo_modes = {row["id"]: row["mode"] for row in hippo_records}

# 3. inspect the first query: its mode, its top paper, and the source text
first_hippo = next(row for row in hippo_records if row["id"] == query_ids[0])
display(
    {
        "mode_counts": {
            mode: list(hippo_modes.values()).count(mode)
            for mode in ("graph_seeded", "dense_fallback")
        },
        "question": queries[query_ids[0]],
        "first_query": first_hippo,
        "first_retrieved_paper": corpus[first_hippo["hipporag"][0]["corpus_id"]],
    }
)

# 4. score both runs with the shared evaluator, side by side
evaluate_run("HippoRAG BGE no-prefix dense", hippo_dense_rankings)
evaluate_run("HippoRAG2", hippo_rankings)
