r"""
### Revisit the same sentences

Only pooling changed. Compare CLS and mean on the same constructed answer,
paraphrase, negation and shared-word distractor from section 03. Which scores
move together? Similarity is not a test of whether a sentence is true.
"""

# 1. stack CLS and mean scores into one 2-row matrix so they share a colour scale
semantic_scores = torch.cat(
    (semantic_cls_scores, util.cos_sim(semantic_mean[:1], semantic_mean[1:]))
)

# 2. draw both pooling rows side by side on the same fitted colour range
fig, ax = plt.subplots(figsize=(8, 3))
sns.heatmap(
    semantic_scores.numpy(),
    annot=True,
    xticklabels=semantic_labels,
    yticklabels=["CLS", "mean"],
    cmap="vlag",
    ax=ax,
)
ax.set_title("Same sentences and weights, different pooling (fitted colour range)")
plt.show()
