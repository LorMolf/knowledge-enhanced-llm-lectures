r"""
### Similar words, different meanings

Four constructed candidates for one question: the answer, a paraphrase, a negation, and a
sentence that shares the words without answering. They are toy sentences on purpose — the
point is the score, not the biology. Read the numbers: an encoder that scores the negation
as high as the answer is measuring topic, not truth.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# one question, four candidates that differ in meaning but not in vocabulary
semantic_texts = [
    "Which protein does compound X bind?",
    "Compound X binds protein Y.",
    "Protein Y is the binding partner of compound X.",
    "Compound X does not bind protein Y.",
    "The study measured compound X in blood.",
]
semantic_labels = ["answer", "paraphrase", "negation", "shared words"]

semantic_cls = encode_biomed(semantic_texts, naive_model, naive_tokenizer)
semantic_cls_scores = util.cos_sim(semantic_cls[:1], semantic_cls[1:]).numpy()[0]

# bars on a zoomed axis: the four scores sit in a 0.02 band, so 0-1 would hide the order
order = semantic_cls_scores.argsort()
fig, ax = plt.subplots(figsize=(6.8, 2.9), constrained_layout=True)
colours = ["#2E7D32" if semantic_labels[i] in ("answer", "paraphrase") else "#9D2235" for i in order]
ax.barh([semantic_texts[1:][i][:46] for i in order], semantic_cls_scores[order], color=colours, height=0.6)
for y, value in enumerate(semantic_cls_scores[order]):
    ax.text(value + 0.0005, y, f"{value:.4f}", va="center", fontsize=7)
ax.set(xlim=(semantic_cls_scores.min() - 0.004, semantic_cls_scores.max() + 0.004),
       xlabel="cosine to the question (axis zoomed to the values)", ylabel="")
ax.tick_params(axis="y", labelsize=7)
ax.set_title("green: the relation holds | red: words without the relation", fontsize=9, loc="left")
plt.show()

print(f"all four scores fall in {semantic_cls_scores.min():.4f} .. {semantic_cls_scores.max():.4f}"
      f" (spread {semantic_cls_scores.max() - semantic_cls_scores.min():.4f})")
for index in order[::-1]:
    print(f"{semantic_cls_scores[index]:.4f}  {semantic_labels[index]:13} {semantic_texts[1:][index]}")
