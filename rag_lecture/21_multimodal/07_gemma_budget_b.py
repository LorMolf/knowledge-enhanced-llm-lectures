r"""
The sweep is assembled into one frame indexed by token budget, so score and encode time sit
side by side. Two panels rather than two y-axes: the top one answers "does the score still
rise", the bottom one answers "what does that cost", and reading them against the same x
axis is the whole trade-off. Where does the curve flatten while the time keeps climbing?
"""

# assemble the budget sweep, one row per token budget
budget_frame = pd.DataFrame(budget_rows).set_index("vision tokens").sort_index()
display(budget_frame[["accuracy@1", "ndcg@10", "encode seconds"]])

# two stacked panels sharing the budget axis: score on top, cost below
fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True, constrained_layout=True,
                                  gridspec_kw={"height_ratios": [1.6, 1]})
top.plot(budget_frame.index, budget_frame["accuracy@1"], marker="o", color=BASELINE, label="accuracy@1")
top.plot(budget_frame.index, budget_frame["ndcg@10"], marker="o", color=OURS, label="nDCG@10")
for column, colour, offset in (("accuracy@1", BASELINE, -11), ("ndcg@10", OURS, 9)):
    top.annotate(f"{budget_frame[column].iloc[-1]:.2f}", (budget_frame.index[-1], budget_frame[column].iloc[-1]),
                 textcoords="offset points", xytext=(-2, offset), fontsize=7, color=colour, ha="right")
top.set(ylim=(0, 1.05), ylabel="score")
top.legend(loc="lower right", ncol=2)
panel(top, "", "", "more pixels per page score better, until they stop paying")

bottom.plot(budget_frame.index, budget_frame["encode seconds"], marker="s", color=MUTED, linestyle="--")
bottom.set(xlabel="vision tokens per page", ylabel="encode seconds")
panel(bottom, "vision tokens per page", "encode seconds")
plt.show()
