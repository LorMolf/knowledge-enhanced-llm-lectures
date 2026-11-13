r"""
### Revisit the same sentences

Only pooling changed. The same four candidates, scored once with the CLS vector and
once with the mean of the token vectors, on one colour scale so the cells are
comparable. Read the negation column: if it moves with the answer column, the score
is measuring topic, not truth.
"""

# two pooling choices, one matrix: stack them so both rows share a colour scale
semantic_mean_scores = util.cos_sim(semantic_mean[:1], semantic_mean[1:]).numpy()[0]
semantic_scores = np.vstack([semantic_cls_scores, semantic_mean_scores])

# draw both rows side by side on the same fitted colour range
fig, ax = plt.subplots(figsize=(7.2, 2.6), constrained_layout=True)
sns.heatmap(semantic_scores, annot=True, fmt=".3f", xticklabels=semantic_labels,
            yticklabels=["CLS", "mean"], cmap="magma", ax=ax,
            vmin=float(semantic_scores.min()), vmax=float(semantic_scores.max()),
            cbar_kws={"label": "cosine"})
ax.set_title("same weights, different pooling: what moved, and did the negation move with it?",
             fontsize=9, loc="left")
plt.show()

print("CLS  :", dict(zip(semantic_labels, semantic_cls_scores.round(3))))
print("mean :", dict(zip(semantic_labels, semantic_mean_scores.round(3))))
print("negation minus answer, CLS:", round(float(semantic_cls_scores[2] - semantic_cls_scores[0]), 3),
      "| mean:", round(float(semantic_mean_scores[2] - semantic_mean_scores[0]), 3))
