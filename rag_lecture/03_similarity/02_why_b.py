r"""
### Counts get the ranking wrong, and the figure shows where

Six candidates for one real question: the passage the dataset labels as the answer, and
five papers that share the question's words. Both scores are proper cosines. Each row is
one paper with a dot per scorer, so the distance between the dots is what the choice of
scorer is worth for that paper, and the labelled row is the one that must come out on top.
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

display(pd.DataFrame({
    "paper": SHORT,
    "shared terms": [overlap(doc_id) for doc_id in CANDIDATES],
    "count cosine": count_cosine.round(3),
    "model cosine": model_cosine.round(3),
    "first 60 chars": [corpus[doc_id][:60].replace("\n", " ") for doc_id in CANDIDATES],
}))

# 5. dumbbell: one row per paper, a dot per scorer, sorted so the model's best is on top
order = np.argsort(model_cosine)
fig, ax = plt.subplots(figsize=(7.6, 3.4), constrained_layout=True)
for row, index in enumerate(order):
    colour = "#9D2235" if index == 0 else "#C9CFD8"
    ax.plot([count_cosine[index], model_cosine[index]], [row, row], color=colour, linewidth=2)
    ax.scatter(count_cosine[index], row, s=42, color="#7F8C9B", zorder=3,
               label="word counts" if row == 0 else None)
    ax.scatter(model_cosine[index], row, s=42, color="#9D2235", zorder=4,
               label="PubMedBERT" if row == 0 else None)
ax.set_yticks(range(len(order)), [SHORT[i] for i in order])
ax.set(xlabel="cosine to the question",
       xlim=(0, 1.15 * max(model_cosine.max(), count_cosine.max())))
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("sorted by the model: the labelled answer is the top row", fontsize=9, loc="left")
plt.show()

# 6. where the labelled answer lands under each ranking, and what beat it
count_rank = int(pd.Series(count_cosine).rank(ascending=False).iloc[0])
model_rank = int(pd.Series(model_cosine).rank(ascending=False).iloc[0])
best_count = SHORT[int(np.argmax(count_cosine))]
print(f"question: {QUESTION[:110]}")
print(f"labelled paper: rank {count_rank} by word counts (beaten by {best_count}), "
      f"rank {model_rank} by the model, of {len(CANDIDATES)}")
print("what word counts put first:", corpus[CANDIDATES[SHORT.index(best_count)]][:160].replace("\n", " "))
