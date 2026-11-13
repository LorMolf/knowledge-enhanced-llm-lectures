r"""
### Three ways to mine the same negative

The table in the previous section reports the held-out test scores; this figure reports what
the three runs actually did. The loss curves are **not** directly comparable: an explicit
negative adds a column, so the same loss value means a different thing in each run. The
validation scores are comparable — same questions, same documents, same evaluator — and the
right panel plots them as dots rather than bars, because three runs landing within a
hundredth of each other is the finding, and identical bars hide it. Do the harder negatives
produce a better validation score, and is the difference larger than one question's worth?
"""

# Two panels: what training did (left), what validation says (right).
fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.4), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1.5, 1]})

# Training loss per run: one line per method, same axes, no smoothing claimed.
for method, history in loss_histories.items():
    frame = pd.DataFrame([row for row in history if "loss" in row])
    left.plot(frame["step"], frame["loss"], marker="o", markersize=3, label=method)
left.legend(frameon=False, loc="upper right")
panel(left, "step", "loss", "training loss (denominators differ, so heights are not a ranking)")

# Validation scores as dots: two metrics per run, annotated, on one fixed scale.
validation = pd.DataFrame(val_scores).T[["ndcg@10", "mrr@10"]]
for row, (method, values) in enumerate(validation.iterrows()):
    right.scatter(values, [row] * len(values), s=44,
                  color=[OURS if metric == "ndcg@10" else BASELINE for metric in validation.columns],
                  edgecolor="white", linewidth=0.6, zorder=3)
    right.annotate(f"{values['ndcg@10']:.3f}", (values["ndcg@10"], row), textcoords="offset points",
                   xytext=(0, 9), fontsize=7, color=OURS, ha="center")
right.set_yticks(range(len(validation)), validation.index, fontsize=8)
right.set(xlim=(validation.min().min() - 0.05, 1.0), ylim=(-0.6, len(validation) - 0.35))
right.scatter([], [], color=OURS, s=40, label="nDCG@10")
right.scatter([], [], color=BASELINE, s=40, label="mrr@10")
right.legend(loc="lower left", ncol=2)

baseline = next(iter(val_scores.values()))["ndcg@10"]
gains = {method: round(values["ndcg@10"] - baseline, 3) for method, values in val_scores.items()}
spread = max(gains.values()) - min(gains.values())
panel(right, "score", "", f"validation, held out\nthe three runs span {spread:.3f}")
plt.show()

print("validation nDCG@10 above the first run:", gains)
print("spread between methods:", round(spread, 3),
      "| one question is worth", round(1 / len(query_ids), 3), "at accuracy@1")
