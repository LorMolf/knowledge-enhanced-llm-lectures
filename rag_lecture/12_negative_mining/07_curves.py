r"""
### Three ways to mine the same negative

The table in the previous section reports the held-out test scores; this figure
reports what the three runs actually did. The loss curves are **not** directly
comparable: an explicit negative adds a column, so the same loss value means a
different thing in each run. The validation bars are comparable, because the
same questions and documents are scored by the same evaluator, and validation is
what may choose between the runs. Do the harder negatives produce a better
validation score, and is the difference larger than one question's worth?
"""

# Two panels: training loss (left) and validation score (right).
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

# 1. Training loss per run — one line per method, same axes for comparison.
for method, history in loss_histories.items():
    frame = pd.DataFrame([row for row in history if "loss" in row])
    axes[0].plot(frame["step"], frame["loss"], marker="o", label=method)
axes[0].set(
    title="Training loss (denominators differ)", xlabel="step", ylabel="loss"
)
axes[0].legend(frameon=False)

# 2. Validation score per run — bars, fixed 0..1 scale so heights are comparable.
pd.DataFrame(val_scores).T.plot.bar(ax=axes[1], ylim=(0, 1), rot=0)
axes[1].set(title="Validation after each run", ylabel="score")

baseline = next(iter(val_scores.values()))["ndcg@10"]
gains = {method: round(values["ndcg@10"] - baseline, 3) for method, values in val_scores.items()}
print("validation nDCG@10 above the first run:", gains)
print("spread between methods:", round(max(gains.values()) - min(gains.values()), 3))

plt.tight_layout()
plt.show()
