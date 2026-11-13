r"""
### The same evaluator, a second ledger

We keep the measuring stick of section 05: the Sentence Transformers evaluator
with the same cutoffs. EmbeddingGemma 2 arrives as a Sentence Transformers model, but the
evaluator expects a ranking either way, so every run follows the section 07 path:
encode, rank with the shared ranking function, then pass the ranked hits to
`compute_metrics`. Scores go to their own
ledger, `mm_rows`, because pages are a different corpus from the BioASQ papers in
the section 20 ledger. Rerunning a model replaces its row.

Before looking at any numbers: with one labelled page among 300, what accuracy@10
would a random ranking reach?
"""

# Freeze the corpus ids once so every model is scored against the same pages.
mm_query_ids = list(mm_queries)
mm_page_ids = list(mm_pages)

# Build the shared evaluator with the section 05 cutoffs; no CSV side effects.
mm_evaluator = InformationRetrievalEvaluator(
    mm_queries,
    mm_pages,
    mm_relevant,
    mrr_at_k=KS,
    ndcg_at_k=KS,
    accuracy_at_k=KS,
    precision_recall_at_k=KS,
    map_at_k=KS,
    write_csv=False,
)

# One ledger for multimodal scores, kept apart from the section 20 rows.
mm_rows = []


def evaluate_mm(name, rankings):
    # Rank order is all the evaluator needs; it computes the metrics itself.
    ordered = [rankings[qid] for qid in mm_evaluator.queries_ids]
    metrics = mm_evaluator.compute_metrics(ordered)

    # Flatten {metric: {k: value}} into one row keyed by "metric@k".
    row = {"model": name}
    row.update(
        {
            f"{metric.replace('@k', '')}@{k}": float(value)
            for metric, values in metrics.items()
            for k, value in values.items()
        }
    )

    # Rerunning a model replaces its row instead of appending a duplicate.
    mm_rows[:] = [existing for existing in mm_rows if existing["model"] != name]
    mm_rows.append(row)
    return row


# Show the ledger is empty and ready before any model is scored.
print(f"evaluator ready: {len(mm_query_ids)} queries, {len(mm_page_ids)} pages, {len(mm_rows)} rows")
