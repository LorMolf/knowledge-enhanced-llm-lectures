r"""
### Score the answers against the reference, honestly

The dataset ships one reference answer per question. We encode it with BGE and
measure the cosine between it and each generated answer. This is a **proxy**: it
rewards topic overlap, not truth, and an on-topic wrong answer scores high. BGE is
both the retriever and the scorer here, and the reference is never retrieved and
never enters an index. What it can show is whether adding evidence moves an answer
towards the reference, and whether that happens for the questions BGE answered badly.

**Question:** does the cosine to the reference rise as we add evidence, and does
the rise depend on whether retrieval found a labelled paper?
"""

# 1. the yardstick, keyed like the questions
references = {str(row["id"]): row["answer"] for row in qa_test}

# 2. encode both sides; the vectors are unit length, so the dot product is the cosine
answer_vectors = bge_model.encode(
    rag_frame["answer"].tolist(), convert_to_tensor=True, normalize_embeddings=True
)
reference_vectors = bge_model.encode(
    [references[qid] for qid in rag_frame["qid"]],
    convert_to_tensor=True,
    normalize_embeddings=True,
)

# 3. cosine per (question, condition) row, stored back on the frame
rag_frame["similarity to reference (proxy)"] = (answer_vectors * reference_vectors).sum(1).cpu()

# 4. one dot per question and condition, split by whether retrieval found anything
per_condition = rag_frame.groupby("condition")["similarity to reference (proxy)"].mean()
print("mean cosine to the reference:", {k: round(v, 3) for k, v in per_condition.items()})
print("delta vs no context:", {k: round(v - per_condition["no context"], 3) for k, v in per_condition.items()})

by_hit = rag_frame.groupby(["condition", "top-3 holds a labelled paper"])["similarity to reference (proxy)"].mean()
print("mean cosine per condition and retrieval outcome:")
print(by_hit.round(3).to_string())

ax = sns.stripplot(
    data=rag_frame,
    x="condition",
    y="similarity to reference (proxy)",
    hue="top-3 holds a labelled paper",
    dodge=True,
    size=3,
)
ax.set(
    ylabel="cosine(answer, reference answer): a proxy",
    title="Same question and prompt, three evidence settings",
)
plt.tight_layout()
plt.show()
