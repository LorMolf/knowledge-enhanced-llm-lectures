r"""
The table above gives three scores per candidate. The figure below draws the same arrows so the geometry behind those numbers is visible: the long tilted vector wins on dot product, while the short aligned one wins on cosine and L2.
"""

# 3. draw the arrows so the geometry behind the scores is visible
fig, ax = plt.subplots(figsize=(5, 4))
for label, vector, color in [
    ("query", toy_query, "black"),
    ("long tilted", toy_docs[0], "tab:blue"),
    ("short aligned", toy_docs[1], "tab:orange"),
]:
    ax.arrow(0, 0, *vector.tolist(), color=color, width=0.015, length_includes_head=True)
    ax.annotate(label, vector.tolist(), xytext=(4, 5), textcoords="offset points")
ax.set(
    xlim=(-0.2, 2.7),
    ylim=(-0.3, 1.5),
    xlabel="dimension 1",
    ylabel="dimension 2",
    title="Illustrative vectors (not model output)",
)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
plt.show()
