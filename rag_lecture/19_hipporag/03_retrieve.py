r"""
### Rank the common test questions

The backend returns original paper IDs, not positions or graph hashes, and a mode for every
question. `dense_fallback` means the native retriever found no facts after reranking and
used dense passage retrieval; `graph_seeded` reports selected facts, not proof that graph
reasoning improved the answer. Compare the native no-prefix dense control and HippoRAG with
the earlier prompted BGE run: all three use the shared evaluator. Can you find the predicted
relation in the source text? Linked papers do not by themselves establish that a question
needs multi-hop reasoning.
"""

# Without a backend response there is nothing to validate; keep the names defined anyway.
hippo_records = hippo_result["results"] if hippo_result else []
hippo_rankings, hippo_dense_rankings, hippo_modes = {}, {}, {}

if not hippo_records:
    print("skipped: no HippoRAG index, because section 14's local model did not start; "
          "the comparison ledger keeps its other rows.")
else:
    # One row per query, no duplicates, and a known mode for each.
    if (
        len(hippo_records) != len(query_ids)
        or len({row["id"] for row in hippo_records}) != len(hippo_records)
        or {row["id"] for row in hippo_records} != set(query_ids)
    ):
        raise ValueError("Backend returned missing, duplicate or unexpected query IDs")
    if any(row["mode"] not in {"dense_fallback", "graph_seeded"} for row in hippo_records):
        raise ValueError("Backend did not identify every retrieval mode")

    hippo_rankings = {row["id"]: row["hipporag"] for row in hippo_records}
    hippo_dense_rankings = {row["id"]: row["dense"] for row in hippo_records}
    hippo_modes = {row["id"]: row["mode"] for row in hippo_records}

    # Inspect the first query: its mode, its top paper, and the source text behind it.
    first_hippo = next(row for row in hippo_records if row["id"] == query_ids[0])
    display({
        "mode_counts": {mode: list(hippo_modes.values()).count(mode)
                        for mode in ("graph_seeded", "dense_fallback")},
        "question": queries[query_ids[0]],
        "first_query": first_hippo,
        "first_retrieved_paper": corpus[first_hippo["hipporag"][0]["corpus_id"]],
    })

    # Score both runs with the shared evaluator, side by side.
    evaluate_run("HippoRAG BGE no-prefix dense", hippo_dense_rankings)
    evaluate_run("HippoRAG2", hippo_rankings)
