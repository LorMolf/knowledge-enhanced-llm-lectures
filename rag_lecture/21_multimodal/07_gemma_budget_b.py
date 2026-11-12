r"""
The sweep is assembled into one frame indexed by token budget, so score and encode time sit side by side. The plot puts score on the left axis and encode seconds on the right, showing where extra tokens stop paying for themselves.
"""

# 4. assemble the budget sweep, one row per token budget
budget_frame = pd.DataFrame(budget_rows).set_index("vision tokens").sort_index()
display(budget_frame[["accuracy@1", "ndcg@10", "encode seconds"]])

# 5. plot score (left axis) and encode time (right axis) against budget
fig, ax = plt.subplots(figsize=(7, 3.5))
for column, label in (("accuracy@1", "accuracy@1"), ("ndcg@10", "nDCG@10")):
    ax.plot(budget_frame.index, budget_frame[column], marker="o", label=label)
ax.set(xlabel="vision tokens per page", ylabel="score", ylim=(0, 1))
time_ax = ax.twinx()
time_ax.plot(
    budget_frame.index, budget_frame["encode seconds"], "k--", marker="s", label="encode seconds"
)
time_ax.set_ylabel("encode seconds")
ax.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.show()
