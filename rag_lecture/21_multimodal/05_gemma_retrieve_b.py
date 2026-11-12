r"""
### Ranking and the first question

Ranking and evaluation are the section 07 calls, now with page IDs. The first question is plotted with its three highest-ranked pages, and the labelled page as a fourth panel so the right answer is always visible. Is the labelled page ranked where you would expect, and what do the pages above it have in common?
"""

# 1. rank pages per query and score the retrieval
gemma_rankings = rank_dense(gemma_query_embeddings, gemma_page_embeddings, mm_query_ids, mm_page_ids)
display(evaluate_mm("EmbeddingGemma 2, 280 image tokens", gemma_rankings))

# 2. show top-3 hits for the first question beside the labelled page
labelled_page = next(iter(mm_relevant[first_qid]))
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
for rank, (ax, hit) in enumerate(zip(axes, gemma_rankings[first_qid][:3]), start=1):
    ax.imshow(mm_pages[hit["corpus_id"]])
    ax.set_title(f"rank {rank}, score {hit['score']:.2f}", fontsize=9)
    ax.axis("off")
axes[3].imshow(mm_pages[labelled_page])
axes[3].set_title("labelled page", fontsize=9)
axes[3].axis("off")
fig.suptitle(mm_queries[first_qid], fontsize=10, wrap=True)
fig.tight_layout()
plt.show()
