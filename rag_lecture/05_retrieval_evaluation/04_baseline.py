r"""
### First real score, and the table it opens

The naive CLS encoder from section 02 is the baseline every later retriever is
measured against. `naive_rankings` was computed there, so this cell only passes it
through `evaluate_run` and shows the ledger. Read a row across the four columns: accuracy@1
and mrr@10 talk about the first hit, recall@10 about how much of the labelled set
came back, and ndcg@10 about where the relevant papers sat in between.
"""

# The baseline row: every later retriever is compared with this one.
naive_metrics = evaluate_run("PubMedBERT CLS", naive_rankings)
display(pd.DataFrame(evaluation_rows).set_index("model"))
print("baseline accuracy@1:", round(naive_metrics["accuracy@k"][1], 3), "| nDCG@10:", round(naive_metrics["ndcg@k"][10], 3))
