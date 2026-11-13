r"""
The reranker is a small published checkpoint (`cross-encoder/ms-marco-MiniLM-L-6-v2`, 22M parameters) trained on general web passages, not biomedical ones. Compare the first paper before and after: does its text support the question better? A reranker improves the order of what the retriever returned; it cannot recover a paper the retriever never found.
"""

# evaluate the reranked run and compare it side by side with plain BGE
reranked_metrics = evaluate_run("BGE + cross-encoder", reranked_rankings)
display(
    pd.DataFrame([row for row in evaluation_rows if row["model"] in ("BGE", "BGE + cross-encoder")])
    .set_index("model")[["accuracy@1", "recall@10", "mrr@10", "ndcg@10"]]
)

# pick one question whose order actually moved and show the new top paper
moved_qid = next(
    (
        qid
        for qid in query_ids
        if [hit["corpus_id"] for hit in all_rankings["BGE"][qid]]
        != [hit["corpus_id"] for hit in reranked_rankings[qid]]
    ),
    query_ids[0],
)
display(
    {
        "question_id": moved_qid,
        "question": queries[moved_qid],
        "BGE order": bge_top_ids[moved_qid],
        "reranked order": [hit["corpus_id"] for hit in reranked_rankings[moved_qid]],
        "new first paper": corpus[reranked_rankings[moved_qid][0]["corpus_id"]][:600],
        "labelled relevant": reranked_rankings[moved_qid][0]["corpus_id"] in relevant_docs[moved_qid],
    }
)

# free the reranker before the next cell
del reranker
