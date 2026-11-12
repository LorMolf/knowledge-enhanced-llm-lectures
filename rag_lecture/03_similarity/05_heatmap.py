r"""
### What do the actual model vectors look like?

Now compute cosine scores from section 02's real CLS vectors, one row per
question. The first five columns are those questions' own labelled papers, so a
diagonal is what a working retriever would give; the last three are ordinary
corpus documents. The colour range is fitted to the observed scores, because raw
CLS cosines sit in a narrow band near the top of the scale. Read the numbers:
is the labelled paper the highest score in its row?
"""

import seaborn as sns

# 1. pick the first labelled paper for each of the first five questions
heat_ids = [sorted(relevant_docs[qid])[0] for qid in query_ids[:5]]

# 2. pad the column set with ordinary corpus docs until we have eight
for doc_id in corpus_ids:
    if len(heat_ids) == 8:
        break
    if doc_id not in heat_ids:
        heat_ids.append(doc_id)

# 3. map those doc ids to their rows in the corpus embedding matrix
heat_positions = [corpus_ids.index(doc_id) for doc_id in heat_ids]

# 4. cosine scores: normalise both sides, then take the dot product
small_scores = (
    F.normalize(naive_query_embeddings[:5], dim=1)
    @ F.normalize(naive_corpus_embeddings[heat_positions], dim=1).T
)

# 5. draw the 5x8 grid, annotating each cell with its score
fig, ax = plt.subplots(figsize=(10, 3))
sns.heatmap(
    small_scores.numpy(),
    annot=True,
    fmt=".2f",
    xticklabels=heat_ids,
    yticklabels=query_ids[:5],
    cmap="vlag",
    center=0,
    ax=ax,
)
ax.set(
    xlabel="document (first five: each question's labelled paper)",
    ylabel="question ID",
    title="Observed CLS cosine similarities",
)
fig.tight_layout()
plt.show()
