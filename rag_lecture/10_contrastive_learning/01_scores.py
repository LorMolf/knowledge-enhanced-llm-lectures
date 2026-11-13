r"""
<a name="section-10"></a>

# Contrastive and metric learning

The raw similarity matrix is the starting point: each row is a query and each column is a positive document, so the diagonal holds the true matches. The heatmap shows which query row has the weakest diagonal relative to its off-diagonal competitors.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# a made-up 3x3 score matrix: rows are queries, columns are positive docs
scores = torch.tensor([[0.8, 0.2, 0.4], [0.3, 0.6, 0.5], [0.1, 0.2, 0.9]])

# draw the raw similarities so we can see which row is ambiguous
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(scores.numpy(), annot=True, vmin=0, vmax=1, cmap="Blues", ax=ax)
ax.set(xlabel="Positive document column", ylabel="Query row", title="Illustrative similarities")
fig.tight_layout()
plt.show()
