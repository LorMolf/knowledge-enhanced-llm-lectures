r"""
### The heatmap

Darker means the first labelled paper came earlier. A row that is blank across every
column is a question this corpus and these retrievers both fail; a row with one dark cell
and nine blanks says something different, about one run rather than about the task.

"""

# Shorten the row labels, or the axis eats half the figure.
labelled_index = first_rank.copy()
labelled_index.index = [f"{qid} · {queries[qid][:50]}" for qid in first_rank.index]

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    labelled_index,
    annot=True,
    fmt=".0f",
    cmap="viridis_r",
    vmin=1,
    vmax=MAX_K,
    cbar_kws={"label": "rank of the first labelled paper"},
    ax=ax,
)
ax.set(xlabel="run", ylabel="question", title=f"Rank of the first labelled paper (blank: none in the top {MAX_K})")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.show()
