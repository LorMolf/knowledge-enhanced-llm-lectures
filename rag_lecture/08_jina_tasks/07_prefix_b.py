r"""
### Rank and compare with the role-aware run

Rank with the same dense retriever used by the role-aware run, then show the ledger metrics and the top-3 side by side. Compare the first ranking and the ledger metrics with the role-aware retrieval run. What changed, and could incomplete relevance labels affect the interpretation?
"""

# Rank with the same dense retriever used by the role-aware run
jina_no_prefix_hits = rank_dense(jina_no_prefix_queries, jina_no_prefix_docs, query_ids)

# Show ledger metrics and the top-3 side by side with the role-aware run
display(
    {
        "no_prefix": evaluate_run("Jina retrieval no prefixes", jina_no_prefix_hits),
        "question": query_texts[0],
        "role_aware_hits": [
            {**hit, "excerpt": corpus[hit["corpus_id"]][:400]}
            for hit in jina_retrieval_rankings[query_ids[0]][:3]
        ],
        "no_prefix_hits": [
            {**hit, "excerpt": corpus[hit["corpus_id"]][:400]}
            for hit in jina_no_prefix_hits[query_ids[0]][:3]
        ],
    }
)

# Free the previous model and embeddings before the next ablation
del (
    jina_model,
    jina_retrieval_docs,
    jina_retrieval_queries,
    jina_no_prefix_docs,
    jina_no_prefix_queries,
)
