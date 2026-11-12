r"""
### Similar words, different meanings

These sentences are **constructed examples**, not biomedical findings. The
first two candidate sentences express the same relation; the third negates it,
and the fourth shares words without answering the question. Let's score them
with the actual CLS encoder. Does a high cosine score distinguish an answer
from a negation or a shared-keyword distractor? Keep these texts for pooling.
"""

# 1. fixed candidate sentences: answer, paraphrase, negation, shared-keyword distractor
semantic_texts = [
    "Which protein does compound X bind?",
    "Compound X binds protein Y.",
    "Protein Y is the binding partner of compound X.",
    "Compound X does not bind protein Y.",
    "The study measured compound X in blood.",
]
semantic_labels = ["answer", "paraphrase", "negation", "shared words"]

# 2. encode every sentence with the CLS encoder so we can compare vectors
semantic_cls = encode_biomed(semantic_texts, naive_model, naive_tokenizer)

# 3. cosine of the question against each candidate
semantic_cls_scores = util.cos_sim(semantic_cls[:1], semantic_cls[1:])

# 4. draw the score matrix so the four candidates sit side by side
fig, ax = plt.subplots(figsize=(8, 2))
sns.heatmap(
    semantic_cls_scores.numpy(),
    annot=True,
    xticklabels=semantic_labels,
    yticklabels=["question"],
    cmap="vlag",
    ax=ax,
)
ax.set_title("Observed CLS scores, colour range fitted to these five sentences")
plt.show()
