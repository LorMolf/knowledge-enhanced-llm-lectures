r"""
### Every run on one axis

The ledger has one row per experiment and each row was measured by the same
evaluator. Sort by nDCG@10 and put every run on one axis: recall@10 says how
much of the labelled set the run found, nDCG@10 says how early it found it. The
gap between the two markers on a row is the difference between "somewhere in the
ten" and "near the top". Which family of changes moves both, and which moves only
one? The table below asks the same question on a single query, where the scores
are paper IDs rather than averages.
"""

import matplotlib.pyplot as plt

# 1. one row per run, ordered by the metric we care about most (nDCG@10)
metric_table = comparison.set_index("model")[["recall@10", "ndcg@10"]].sort_values("ndcg@10")

# 2. draw both metrics on the same axis so the gap per run is visible
fig, ax = plt.subplots(figsize=(8, 0.3 * len(metric_table) + 1.8))
for column, marker in (("recall@10", "o"), ("ndcg@10", "s")):
    ax.scatter(metric_table[column], metric_table.index, marker=marker, s=30, label=column)
ax.set(xlabel="score", xlim=(0, 1), title=f"{len(metric_table)} runs, sorted by nDCG@10")
ax.grid(axis="x", alpha=0.3)
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.show()

# 3. drill into one query: what does each system put at rank 1?
comparison_qid = query_ids[0]
print("Question:", comparison_qid, queries[comparison_qid])

# One row per system: the paper it put first, whether that paper is labelled, its text.
SYSTEMS = ("BM25", "BGE", "BM25+BGE RRF")
top_rows = [
    {
        "system": name,
        "rank-1 paper": all_rankings[name][comparison_qid][0]["corpus_id"],
        "labelled for this question": all_rankings[name][comparison_qid][0]["corpus_id"]
        in relevant_docs[comparison_qid],
        "excerpt": corpus[all_rankings[name][comparison_qid][0]["corpus_id"]][:160],
    }
    for name in SYSTEMS
]
display(pd.DataFrame(top_rows))
