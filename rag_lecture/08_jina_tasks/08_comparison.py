r"""
### Compare the task settings together

Here are six retrieval measures at cutoff 10, all from the shared evaluator.
Changing the adapter and removing role prefixes are separate interventions.
Which setting changes the first-hit measures, and which changes multi-paper
recall? The complete ledger also retains the other cutoffs.
"""

# keep only the Jina rows, indexed by model name
jina_table = pd.DataFrame(evaluation_rows).set_index("model").filter(like="Jina", axis=0)

# the six cutoff-10 measures we compare side by side
jina_columns = [
    f"{metric}@10" for metric in ("accuracy", "precision", "recall", "mrr", "ndcg", "map")
]

# one heatmap so every setting is read on the same 0..1 scale
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(jina_table[jina_columns], annot=True, fmt=".3f", vmin=0, vmax=1, cmap="Blues", ax=ax)
ax.set_title("Observed Jina task and role ablations")
fig.tight_layout()
plt.show()

# show the underlying numbers the heatmap colors encode
print(jina_table[jina_columns].round(3))
