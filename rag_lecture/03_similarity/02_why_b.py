r"""
### Counts get the ranking wrong, and the figure shows where

Six candidates for one real question: the passage the dataset labels as the answer, and
five papers that share the question's words. Both scores are proper cosines — divided by
both norms — and both ranks are printed, so nothing here is inferred from a bar height.
"""

# 101 continued: same arithmetic, two sets of numbers, two rankings.
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QID = query_ids[0]
QUESTION = query_texts[0]
LABELLED = sorted(relevant_docs[QID])[0]

# 1. five papers that share the question's words without being the labelled answer
question_terms = {word for word in re.findall(r"[a-z]{4,}", QUESTION.lower())} - STOP


def overlap(doc_id):
    return len(set(re.findall(r"[a-z]{4,}", corpus[doc_id].lower())) & question_terms)


distractors = sorted((d for d in corpus_ids if d != LABELLED), key=overlap, reverse=True)[:5]
CANDIDATES = [LABELLED] + distractors
SHORT = ["labelled"] + [f"D{i}" for i in range(1, 6)]

# 2. count vectors over the question's vocabulary, matched on word boundaries
counts = pd.DataFrame(
    [[len(re.findall(rf"\b{word}\b", corpus[doc_id].lower())) for word in sorted(question_terms)]
     for doc_id in CANDIDATES],
    index=SHORT, columns=sorted(question_terms),
)
query_counts = np.array(
    [len(re.findall(rf"\b{word}\b", QUESTION.lower())) for word in sorted(question_terms)], float)

# 3. a cosine is a dot product over both norms: one pair, written out
a, b = counts.loc["labelled"].to_numpy(float), counts.loc["D1"].to_numpy(float)
print("labelled vector:", a.astype(int), "| D1 vector:", b.astype(int))
print("labelled . D1 =", float(a @ b),
      "| |labelled| =", round(float(np.linalg.norm(a)), 2),
      "| |D1| =", round(float(np.linalg.norm(b)), 2))

# 4. two rankings over the same six candidates, both as normalised cosines
count_matrix = counts.to_numpy(float)
count_cosine = (count_matrix @ query_counts) / (
    np.linalg.norm(count_matrix, axis=1) * np.linalg.norm(query_counts) + 1e-9)

doc_vectors = naive_corpus_embeddings[[corpus_ids.index(d) for d in CANDIDATES]].numpy()
query_vector = naive_query_embeddings[0:1].numpy()
model_cosine = (
    (doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True))
    @ (query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)).T
).squeeze(1)

table = pd.DataFrame({
    "paper": SHORT,
    "shared terms": [overlap(doc_id) for doc_id in CANDIDATES],
    "count cosine": count_cosine.round(3),
    "model cosine": model_cosine.round(3),
})
display(table)

# 5. one figure, two series, every value printed
order = np.argsort(-model_cosine)
fig, ax = plt.subplots(figsize=(7.4, 3.4), constrained_layout=True)
x = np.arange(len(order))
ax.bar(x - 0.2, count_cosine[order], 0.4, label="word counts", color="#B9C4D2")
ax.bar(x + 0.2, model_cosine[order], 0.4, label="PubMedBERT", color="#9D2235")
ax.set_xticks(x, [SHORT[i] for i in order])
ax.set(ylabel="cosine to the question", ylim=(0, 1.05))
ax.legend(frameon=False, fontsize=8)
for xi, value in zip(x, model_cosine[order]):
    ax.text(xi + 0.2, value + 0.02, f"{value:.2f}", ha="center", fontsize=7)
for xi, value in zip(x, count_cosine[order]):
    ax.text(xi - 0.2, value + 0.02, f"{value:.2f}", ha="center", fontsize=7, color="#555555")
ax.set_title("the labelled answer is the first bar; where does each scorer put it?", fontsize=9, loc="left")
plt.show()

# 6. where the labelled answer actually lands under each ranking
count_rank = int(pd.Series(count_cosine).rank(ascending=False).iloc[0])
model_rank = int(pd.Series(model_cosine).rank(ascending=False).iloc[0])
print(f"question: {QUESTION[:110]}")
print(f"labelled paper: rank {count_rank} by word counts, rank {model_rank} by the model, of {len(CANDIDATES)}")
