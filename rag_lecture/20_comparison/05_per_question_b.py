r"""
### Where the runs actually differ

One rank per question per run is a wall of numbers, and a hundred-row heatmap is
unreadable on a projector. The informative view is a curve: for each k, the share of
questions whose first labelled paper is in the top k. Two runs that separate here
separate for the whole corpus, not for one lucky question.
"""

import matplotlib.pyplot as plt
import pandas as pd

KS_SHOWN = (1, 3, 5, 10)
curves = pd.DataFrame(
    {run: [(first_rank[run] <= k).mean() for k in KS_SHOWN] for run in first_rank.columns},
    index=KS_SHOWN,
)

fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)
for run in curves.columns:
    ax.plot(curves.index, curves[run], marker="o", linewidth=1.6, label=run)
ax.set(xticks=KS_SHOWN, xlabel="k", ylim=(0, 1.02),
       ylabel="share of questions with a labelled paper in the top k")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("how far down the list the first right paper sits", fontsize=9, loc="left")
plt.show()

display((curves * 100).round(1).astype(str) + "%")
print("questions with no labelled paper in the top", MAX_K, "for at least one run:",
      int(first_rank.isna().any(axis=1).sum()), "of", len(first_rank))
