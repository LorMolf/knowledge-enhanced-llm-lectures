r"""
### What does the cutoff change?

nDCG at four cutoffs, with a bootstrap band over the test questions: the band is the honest
part, because 32 questions cannot resolve a difference smaller than the band's width. The
dashed line is the best any ranking could reach on these labels — labelled papers first,
then the rest — scored by the same evaluator. The gap between the two curves is how much
retrieval quality is still on the table; the table underneath says the same in numbers.
"""

# Build the ideal ranking: labelled papers first, then everything else.
ideal_rankings = {}
for qid in query_ids:
    ordered = sorted(relevant_docs[qid]) + [
        doc_id for doc_id in corpus_ids if doc_id not in relevant_docs[qid]
    ]
    ideal_rankings[qid] = [
        {"corpus_id": doc_id, "score": float(MAX_K - rank)}
        for rank, doc_id in enumerate(ordered[:MAX_K])
    ]

# Score the ideal ranking with the same evaluator as the real run.
ideal_metrics = evaluator.compute_metrics(
    [ideal_rankings[qid] for qid in evaluator.queries_ids]
)


# evaluate_run returns a flat row, compute_metrics nests {metric@k: {k: value}}.
def score(metrics, metric, k):
    return metrics[f"{metric}@{k}"] if f"{metric}@{k}" in metrics else metrics[f"{metric}@k"][k]


# Per-question nDCG@k, so the band is over questions and not over one aggregate.
def ndcg_at_k(hits, relevant, k):
    gains = [1 / np.log2(rank + 1) for rank, hit in enumerate(hits[:k], 1)
             if hit["corpus_id"] in relevant]
    ideal_gains = [1 / np.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1)]
    return sum(gains) / sum(ideal_gains) if ideal_gains else 0.0


per_question = {
    k: np.array([ndcg_at_k(all_rankings["PubMedBERT CLS"][qid], relevant_docs[qid], k)
                 for qid in query_ids])
    for k in KS
}

# The curve, with the band and the ceiling on the same axes.
fig, ax = plt.subplots(constrained_layout=True)
means = np.array([per_question[k].mean() for k in KS])
low = np.array([bootstrap_ci(per_question[k], seed=SEED)[1] for k in KS])
high = np.array([bootstrap_ci(per_question[k], seed=SEED)[2] for k in KS])
ax.fill_between(KS, low, high, color=OURS, alpha=0.16, linewidth=0)
ax.plot(KS, means, color=OURS, marker="o")
ax.plot(KS, [score(ideal_metrics, "ndcg", k) for k in KS], color=BASELINE, linestyle="--")
ax.annotate("ideal", (KS[-1], 1.0), textcoords="offset points", xytext=(-4, 6),
            fontsize=7, color=BASELINE, ha="right")
ax.annotate("PubMedBERT CLS", (KS[-1], means[-1]), textcoords="offset points", xytext=(-4, 10),
            fontsize=7, color=OURS, ha="right")
ax.set(xticks=KS, ylim=(0, 1.08))
panel(ax, "cutoff k", "nDCG@k", "retrieval quality against the best achievable ranking\n95% bootstrap band over 32 questions")
plt.show()

# The other metrics are numbers, not panels: observed against the ceiling at k=10.
comparison = pd.DataFrame({"metric": ["accuracy@1", "recall@10", "ndcg@10", "map@10"]})
comparison["observed"] = [round(score(naive_metrics, m.split("@")[0], 10), 3) for m in comparison.metric]
comparison["ideal"] = [round(score(ideal_metrics, m.split("@")[0], 10), 3) for m in comparison.metric]
comparison["gap"] = (comparison["ideal"] - comparison["observed"]).round(3)
display(comparison)
print("on 32 questions, one question moves accuracy@1 by", round(1 / len(query_ids), 3))
