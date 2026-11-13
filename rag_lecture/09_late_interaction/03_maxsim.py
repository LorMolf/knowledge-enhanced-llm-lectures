r"""
### Which document vectors win, and how often

The full MaxSim matrix is 32 query vectors by a few hundred document vectors: at that size
every cell is a pixel and the figure teaches nothing. So the arithmetic is answered in
numbers first — how many distinct document vectors win at least one row, and which ones are
reused — and the picture zooms into the slice that carries the answer: the ten columns that
win most often, twelve query vectors, values in every cell, and a green box on the column
each row keeps. Where a column collects several boxes, that is one document vector doing the
work for many query vectors.

MaxSim is not the ranking: this is the mechanism, the native scorer produces the score.
"""

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

# the readable slice: the ten columns that win most, the first twelve query vectors
winner_counts = {}
for row, column in enumerate(best_positions.tolist()):
    winner_counts.setdefault(column, []).append(row)
shown_columns = [column for column, _ in sorted(winner_counts.items(), key=lambda pair: -len(pair[1]))][:10]
shown_rows = list(range(min(12, similarities.shape[0])))
slice_scores = similarities[shown_rows][:, shown_columns]

fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
ax.imshow(slice_scores.numpy(), cmap="magma", aspect="auto")
for y, row in enumerate(shown_rows):
    for x, column in enumerate(shown_columns):
        value = float(slice_scores[y, x])
        ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=6,
                color="black" if value > float(slice_scores.max()) * 0.55 else "white")
    best = int(slice_scores[y].argmax())
    ax.add_patch(plt.Rectangle((best - 0.5, y - 0.5), 1, 1, fill=False,
                               edgecolor="#7ee787", linewidth=1.4))
ax.set(xticks=range(len(shown_columns)), xticklabels=[f"d{column}" for column in shown_columns],
       yticks=range(len(shown_rows)), yticklabels=[f"q{row}" for row in shown_rows],
       xlabel="document vectors that win most often", ylabel="query vectors")
panel(ax, "document vectors that win most often", "query vectors",
      "the slice that carries the answer: one green box per query vector")
plt.show()
