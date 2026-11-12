r"""
### Score a trained model, on validation first

Validation is the only signal that may choose between the three runs; the test set
is reported afterwards, for the record, and never used to pick a winner. The corpus
is encoded once per trained model, because the document side changes when the
weights change. Both rankings go through the same `rank_dense` used everywhere else,
so a difference here is the model, not the pipeline.
"""

def score_one(method, model):
    """Encode the corpus once, then score validation and test for this run."""
    docs = model.encode(corpus_texts, convert_to_tensor=True, device="cuda").cpu()

    # 1. validation: question side only, with the published query prompt
    val_vectors = model.encode(
        list(val_queries.values()), prompt=BGE_QUERY_PROMPT, convert_to_tensor=True, device="cuda"
    ).cpu()
    val_hits = rank_dense(val_vectors, docs, qids=list(val_queries))
    val_metrics = validation_evaluator.compute_metrics(
        [val_hits[qid] for qid in validation_evaluator.queries_ids]
    )

    # 2. test: same encoding path, scored into the shared ledger
    test_vectors = model.encode(
        query_texts, prompt=BGE_QUERY_PROMPT, convert_to_tensor=True, device="cuda"
    ).cpu()
    test_metrics = evaluate_run(
        f"BGE {method} negative", rank_dense(test_vectors, docs, qids=query_ids)
    )
    return val_metrics, test_metrics


print("score_one defined: corpus encoded once, validation and test scored separately.")
