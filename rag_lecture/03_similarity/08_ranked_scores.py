r"""
### Inspect the ranked scores

The order matters, but so does the score pattern. Let's mark the first
question's judged relevant papers in its actual top ten. A small score gap is
not a probability, and an unjudged paper is not necessarily irrelevant.
Which paper would you inspect first to understand a retrieval error?
"""

# 1. pull the ids and scores out of the hit dicts, in rank order
ranked_ids = [hit["corpus_id"] for hit in first_hits]
ranked_scores = [hit["score"] for hit in first_hits]

# 2. color each bar by whether the judged qrels marked it relevant
colors = [
    "tab:green" if doc_id in relevant_docs[query_ids[0]] else "tab:gray" for doc_id in ranked_ids
]

# 3. draw a lollipop chart: one stem + dot per ranked document
fig, ax = plt.subplots(figsize=(10, 3))
ax.vlines(range(1, len(first_hits) + 1), min(ranked_scores), ranked_scores, color=colors, alpha=0.5)
ax.scatter(range(1, len(first_hits) + 1), ranked_scores, color=colors, zorder=3)

# 4. fit the y-axis to the score band so small gaps stay visible
ax.set_ylim(min(ranked_scores) - 0.02, max(ranked_scores) + 0.02)

# 5. label each position with its document id, in rank order
ax.set_xticks(range(1, len(first_hits) + 1), ranked_ids, rotation=45)
ax.set(
    xlabel="document ID, in rank order",
    ylabel="cosine similarity",
    title="Actual top ten, fitted scale: green = judged relevant; gray = unjudged",
)
fig.tight_layout()
plt.show()
