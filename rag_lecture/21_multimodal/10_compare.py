r"""
### Compare the encoders

Every row uses the same 100 questions, the same 300 pages and the same evaluator.
The models differ in more than quality: EmbeddingGemma 2 loads about 440M
parameters for text and images and stores 768 numbers per page, MetaEncoder v1
has 28.4B parameters and 6,656 numbers per page. In float32 that is about 3.1 GB
against 26.6 GB of vectors per million pages. Random ranking would put accuracy@10
near 3%. With only 100 questions drawn from 65 pages, which gaps would you trust,
and which cost would decide for your use case? The MetaEncoder rows appear only
where the GPU was large enough to run the pair before this one.
"""

# 1. Assemble the per-model results into one table, indexed by model name
mm_table = pd.DataFrame(mm_rows).set_index("model")

# 2. Show the retrieval metrics side by side so gaps are easy to compare
display(mm_table[["accuracy@1", "accuracy@3", "accuracy@10", "mrr@10", "ndcg@10"]])

# 3. Plot the headline metrics as horizontal bars, one bar group per model
ax = mm_table[["accuracy@1", "mrr@10", "ndcg@10"]].plot.barh(figsize=(8, 4))
ax.set(xlabel="mean score", xlim=(0, 1), title="Question to page retrieval, 300-page corpus")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
