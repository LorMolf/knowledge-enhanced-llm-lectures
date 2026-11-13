r"""
<a name="section-03"></a>

# Similarity, geometry and rankings

Before any model, a similarity is arithmetic on counts. Six real abstracts from the
corpus, a vocabulary of the eight words they share most often, and one dot product
written out by hand. If that number is clear, every vector in the rest of the lecture
is the same operation with a better set of numbers.
"""

# 101: a similarity is a number you can compute with pen and paper.
import re
from collections import Counter

import numpy as np
import pandas as pd

TEXTS = corpus_texts[:6]
SHORT = [f"P{i}" for i in range(1, 7)]
STOP = {"with", "that", "this", "from", "were", "have", "been", "their", "which", "these"}

tally = Counter(word for text in TEXTS for word in re.findall(r"[a-z]{4,}", text.lower()))
VOCAB = [word for word, _ in tally.most_common(12) if word not in STOP][:8]

counts = pd.DataFrame(
    [[text.lower().count(word) for word in VOCAB] for text in TEXTS],
    index=SHORT, columns=VOCAB,
)
print("vocabulary:", ", ".join(VOCAB))
display(counts)

# one pair by hand: the dot product is a sum of products, nothing else
a, b = counts.loc["P1"].to_numpy(), counts.loc["P2"].to_numpy()
terms = " + ".join(f"{x}x{y}" for x, y in zip(a, b) if x and y)
print("P1:", a, "\nP2:", b)
print("P1 . P2 =", terms or "0", "=", int(a @ b))
print("|P1| =", round(float(np.linalg.norm(a)), 2), "|P2| =", round(float(np.linalg.norm(b)), 2))
print("cosine(P1, P2) =", round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)), 3))
