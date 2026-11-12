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


# 4. One panel: nDCG at each cutoff, observed against the best possible ranking.
fig, ax = plt.subplots(figsize=(6.6, 3.2), constrained_layout=True)
ax.plot(KS, [score(naive_metrics, "ndcg", k) for k in KS], marker="o", label="PubMedBERT CLS")
ax.plot(KS, [score(ideal_metrics, "ndcg", k) for k in KS], linestyle="--", color="gray",
        label="ideal ranking")
ax.set(xticks=KS, xlabel="cutoff k", ylabel="nDCG@k", ylim=(0, 1))
ax.legend(frameon=False, fontsize=8)
ax.set_title("how much of the achievable nDCG this ranking reaches", fontsize=9, loc="left")
plt.show()

# 5. the other metrics are numbers, not panels: observed against the ceiling at k=10.
comparison = pd.DataFrame({
    "metric": ["accuracy@1", "recall@10", "ndcg@10", "map@10"],
})
comparison["observed"] = [round(score(naive_metrics, m.split("@")[0], 10), 3) for m in comparison.metric]
comparison["ideal"] = [round(score(ideal_metrics, m.split("@")[0], 10), 3) for m in comparison.metric]
comparison["gap"] = (comparison["ideal"] - comparison["observed"]).round(3)
display(comparison)
print("on 32 questions, one question moves accuracy@1 by", round(1 / len(query_ids), 3))
