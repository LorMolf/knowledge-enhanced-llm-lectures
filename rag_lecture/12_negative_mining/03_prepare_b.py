r"""
The strip plot below puts every chosen negative on the same BGE scale as the
labelled positives: the closer a mined negative sits to a positive's cosine, the
harder the example, and the more it matters that it really is not relevant.
"""

# one ID -> column map, so the scoring below does not scan corpus_ids per row
position = {doc_id: column for column, doc_id in enumerate(corpus_ids)}

# score every negative and every labelled positive on the same BGE scale
hardness = [
    {"pair": method, "cosine": float(dense_mining_scores[i, position[doc_id]])}
    for method, ids in negative_ids.items()
    for i, doc_id in enumerate(ids)
] + [
    {
        "pair": "labelled positive",
        "cosine": float(
            dense_mining_scores[i, position[selected_positive_ids[qid]]]
        ),
    }
    for i, qid in enumerate(train_qids)
]

# strip plot: how close each negative sits to its question vs the positive
fig, ax = plt.subplots(figsize=(7, 3))
sns.stripplot(pd.DataFrame(hardness), x="pair", y="cosine", size=3, ax=ax)
ax.set(title="How hard is each negative, in BGE cosine", ylabel="cosine(question, document)")
plt.tight_layout()
plt.show()
