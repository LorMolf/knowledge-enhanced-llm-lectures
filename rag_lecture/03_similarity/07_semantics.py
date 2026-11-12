r"""
### Similar words, different meanings

Four constructed candidates for one question: the answer, a paraphrase, a negation, and a
sentence that shares the words without answering. They are toy sentences on purpose — the
point is the score, not the biology. Read the numbers: an encoder that scores the negation
as high as the answer is measuring topic, not truth.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# 1. one question, four candidates that differ in meaning but not in vocabulary
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

# 2. the four scores as bars: one scale, values printed, no colour legend to decode
fig, ax = plt.subplots(figsize=(6.4, 2.8), constrained_layout=True)
order = semantic_cls_scores.argsort()[::-1]
colours = ["#2E7D32" if semantic_labels[i] in ("answer", "paraphrase") else "#9D2235" for i in order]
ax.barh([semantic_texts[1:][i][:46] for i in order], semantic_cls_scores[order], color=colours, height=0.6)
for y, value in enumerate(semantic_cls_scores[order]):
    ax.text(value + 0.01, y, f"{value:.3f}", va="center", fontsize=8)
ax.set(xlim=(0, 1), xlabel="cosine to the question", ylabel="")
ax.tick_params(axis="y", labelsize=7)
ax.set_title("green: the relation holds | red: words without the relation", fontsize=9, loc="left")
plt.show()

for label, text, score in sorted(zip(semantic_labels, semantic_texts[1:], semantic_cls_scores), key=lambda row: -row[2]):
    print(f"{score:.3f}  {label:13} {text}")
