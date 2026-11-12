r"""
### See the token similarities

For the first question and one of its labelled-positive documents, multiply their normalized token-vector matrices. Each row corresponds to a query token vector and each column to a document token vector after the checkpoint's scoring mask. Query expansion and special tokens can appear among retained vectors; indices are not verified word or tokenizer positions. The orange maximum in each row contributes to the sum. This calculation explains the mechanism; the ranking uses the native scorer.

Which retained document vectors are selected by multiple query vectors?
"""

import matplotlib.pyplot as plt
import seaborn as sns

# 1. pick one labelled-positive document to inspect
inspect_doc_id = sorted(relevant_docs[query_ids[0]])[0]
inspect_doc_index = corpus_ids.index(inspect_doc_id)
print("Question ID and text:", query_ids[0], query_texts[0])
print("Labelled document ID and excerpt:", inspect_doc_id, corpus[inspect_doc_id][:600])

# 2. score every query vector against every document vector (MaxSim matrix)
similarities = (
    torch.as_tensor(colbert_queries[0]).cpu()
    @ torch.as_tensor(colbert_docs[inspect_doc_index]).cpu().T
)

# 3. for each query vector, find the document vector it matches best
best_positions = similarities.argmax(dim=1)

# 4. draw the matrix and mark each row's maximum in orange
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(similarities.numpy(), ax=ax, cmap="viridis")
ax.scatter(
    best_positions.numpy() + 0.5,
    torch.arange(len(best_positions)).numpy() + 0.5,
    marker="x",
    color="orange",
    s=12,
)
ax.set(
    xlabel="Retained document-vector index",
    ylabel="Retained query-vector index",
    title="ColBERT token similarities and row maxima",
)
plt.show()
