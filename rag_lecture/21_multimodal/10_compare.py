r"""
### What the multimodal encoder buys you

Every row uses the same 100 questions, the same 300 pages and the same evaluator. The
vision rows differ in how many pixels the model may look at; the OCR row is a different
pipeline into the same model, which is the comparison that decides whether pixels earn
their place at all.
EmbeddingGemma 2 loads about 440M parameters and stores 768 numbers per page: 3.1 GB of
vectors per million pages in float32. Random ranking would put accuracy@10 near 3%, which is
the line to beat, and with 100 questions drawn from 65 annotated pages the gap between two
runs is worth less than one question moving. Which cutoffs does the image tower win, and
where does the text-only baseline still hold?
"""

# One table, indexed by run, so the metric columns sit side by side.
mm_table = pd.DataFrame(mm_rows).set_index("model")

# The numbers first, because three significant figures beat a bar chart.
display(mm_table[["accuracy@1", "accuracy@3", "accuracy@10", "mrr@10", "ndcg@10"]].round(3))

# One row per run, one dot per metric: the ranking is the reading, not the bar area.
metrics_shown = ["accuracy@1", "ndcg@10"]
fig, ax = plt.subplots(constrained_layout=True)
for row, (name, values) in enumerate(mm_table[metrics_shown].iterrows()):
    ax.scatter(values, [row] * len(metrics_shown), s=46,
               color=[OURS if metric == "ndcg@10" else BASELINE for metric in metrics_shown],
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.annotate(f"nDCG {values['ndcg@10']:.3f}", (values["ndcg@10"], row),
                textcoords="offset points", xytext=(0, 9), fontsize=7, color=OURS, ha="center")
ax.axvline(0.03, color=ACCENT, linestyle=":", linewidth=1)
ax.annotate("random", (0.03, -0.45), textcoords="offset points", xytext=(3, 0),
            fontsize=7, color=ACCENT)
ax.set_yticks(range(len(mm_table)), mm_table.index, fontsize=8)
ax.set(xlim=(0, 1), ylim=(-0.6, len(mm_table) - 0.35))
ax.scatter([], [], color=OURS, s=40, label="nDCG@10")
ax.scatter([], [], color=BASELINE, s=40, label="accuracy@1")
ax.legend(loc="lower right")
panel(ax, "score", "", "page-image retrieval, 300 pages and 100 questions\nsame labels and evaluator for every row")
plt.show()
