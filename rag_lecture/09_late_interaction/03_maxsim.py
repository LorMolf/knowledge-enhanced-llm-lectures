r"""
### Which document vectors win, and how often

The MaxSim matrix is 32 rows by a few hundred columns: the values matter, the axis
ticks do not. So the matrix stays unlabelled, and the question is answered in numbers
instead — how many distinct document vectors win at least one row, and which ones win
more than once. A query vector that matches nothing strongly, and a document vector that
several query vectors all pick, are the two shapes worth recognising.

MaxSim is not the ranking: this is the mechanism, the native scorer produces the score.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch

# pick one labelled-positive document to inspect
inspect_doc_id = sorted(relevant_docs[query_ids[0]])[0]
inspect_doc_index = corpus_ids.index(inspect_doc_id)
print("question:", query_texts[0][:120])
print("labelled paper:", inspect_doc_id, "|", corpus[inspect_doc_id][:120])

# every query vector against every retained document vector
similarities = (
    torch.as_tensor(colbert_queries[0]).cpu()
    @ torch.as_tensor(colbert_docs[inspect_doc_index]).cpu().T
)
best_positions = similarities.argmax(dim=1)

# the answer the markdown asks for, before the picture
wins = torch.bincount(best_positions, minlength=similarities.shape[1])
top_winners = wins.topk(min(5, int((wins > 0).sum()))).indices.tolist()
print(f"matrix: {similarities.shape[0]} query vectors x {similarities.shape[1]} document vectors")
print("distinct document vectors that win at least one row:", int((wins > 0).sum()))
print("most-reused document vectors (column: number of query vectors):",
      {int(column): int(wins[column]) for column in top_winners})
print("MaxSim score for this pair:", round(float(similarities.max(dim=1).values.sum()), 3))

fig, ax = plt.subplots(figsize=(7.6, 3.4), constrained_layout=True)
sns.heatmap(similarities.numpy(), ax=ax, cmap="magma", cbar_kws={"label": "cosine"})
ax.scatter(best_positions.numpy() + 0.5, torch.arange(len(best_positions)).numpy() + 0.5,
           marker="x", color="#7ee787", s=10, linewidths=0.8)
ax.set(xticks=[], yticks=[], xlabel="retained document vectors (in order)",
       ylabel="query vectors", title="each row picks one document vector; green x marks the pick")
plt.show()
