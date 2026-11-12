r"""
### What does the cutoff change?

Plot all six observed effectiveness measures at the same cutoffs for the initial
run, with a dashed line for the best any ranking could reach on these labels. The
dashed curve is an ideal ordering scored by the same evaluator: labelled papers
first, then the rest. Precision and MAP sit far below it because most questions
have only one or two labelled papers, so precision at ten is capped near 0.1.
Which metric is measuring the retriever, and which is measuring the labels?
"""

import matplotlib.pyplot as plt

# 1. Build the ideal ranking: labelled papers first, then everything else.
ideal_rankings = {}
for qid in query_ids:
    ordered = sorted(relevant_docs[qid]) + [
        doc_id for doc_id in corpus_ids if doc_id not in relevant_docs[qid]
    ]
    ideal_rankings[qid] = [
        {"corpus_id": doc_id, "score": float(MAX_K - rank)}
        for rank, doc_id in enumerate(ordered[:MAX_K])
    ]

# 2. Score the ideal ranking with the same evaluator as the real run.
ideal_metrics = evaluator.compute_metrics(
    [ideal_rankings[qid] for qid in evaluator.queries_ids]
)

# 3. evaluate_run hands back a flat row, compute_metrics nests {metric@k: {k: value}};
#    read either shape so the cell does not depend on which producer filled it.
def score(metrics, metric, k):
    return metrics[f"{metric}@{k}"] if f"{metric}@{k}" in metrics else metrics[f"{metric}@k"][k]


# 4. One panel per metric, observed vs ideal, so the gap is visible.
fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharey=True)
for ax, metric in zip(axes.flat, ("accuracy", "precision", "recall", "mrr", "ndcg", "map")):
    ax.plot(KS, [score(naive_metrics, metric, k) for k in KS], marker="o", label="PubMedBERT CLS")
    ax.plot(
        KS,
        [score(ideal_metrics, metric, k) for k in KS],
        linestyle="--",
        color="gray",
        label="ideal ranking",
    )
    ax.set(title=metric.upper(), xlabel="cutoff k", ylabel="mean score", ylim=(0, 1), xticks=KS)
axes[0, 0].legend(frameon=False, fontsize=8)
gaps = {metric: score(ideal_metrics, metric, 10) - score(naive_metrics, metric, 10)
        for metric in ("accuracy", "recall", "ndcg", "map")}
print("remaining gap at k=10:", {m: round(g, 3) for m, g in gaps.items()})
fig.tight_layout()
plt.show()
