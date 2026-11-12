r"""
### Counts get the ranking wrong, and one figure shows where

Same six papers, one real question. Ranking by the count cosine puts the paper that
repeats the question's words first; ranking by the model's cosine puts the paper that
answers it first. Both scores are printed for the same six documents, so the difference
is the model, not the arithmetic.
"""

# 101 continued: identical arithmetic, two sets of numbers, two rankings.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

QUESTION = query_texts[0]
count_query = np.array([QUESTION.lower().count(word) for word in VOCAB], dtype=float)
count_scores = counts.to_numpy() @ count_query
count_scores /= np.linalg.norm(counts.to_numpy(), axis=1) * np.linalg.norm(count_query) + 1e-9

model_query = naive_query_embeddings[0]
model_scores = (naive_corpus_embeddings[:6] @ model_query).cpu().numpy()

comparison = pd.DataFrame(
    {"paper": SHORT, "text": [text[:60] + "..." for text in TEXTS],
     "count cosine": count_scores.round(3), "model cosine": model_scores.round(3)}
).sort_values("model cosine", ascending=False)
display(comparison)

fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
order = np.argsort(-model_scores)
x = np.arange(len(order))
ax.bar(x - 0.2, count_scores[order], 0.4, label="word counts", color="#B9C4D2")
ax.bar(x + 0.2, model_scores[order], 0.4, label="PubMedBERT", color="#9D2235")
ax.set_xticks(x, [SHORT[i] for i in order])
ax.set(ylabel="cosine to the question", ylim=(0, 1.05))
ax.legend(frameon=False, loc="upper right", fontsize=8)
for xi, value in zip(x, model_scores[order]):
    ax.text(xi + 0.2, value + 0.02, f"{value:.2f}", ha="center", fontsize=7)
for xi, value in zip(x, count_scores[order]):
    ax.text(xi - 0.2, value + 0.02, f"{value:.2f}", ha="center", fontsize=7, color="#555555")
ax.set_title("the answer paper is not the one that repeats the question's words", fontsize=9, loc="left")
plt.show()

best, second = comparison.iloc[0], comparison.iloc[1]
print(f"question: {QUESTION}")
print(f"model ranks {best.paper} first ({best['model cosine']:.3f}); counts rank it {list(comparison.paper).index(best.paper) + 1}")
