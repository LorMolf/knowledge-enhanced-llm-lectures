r"""
### What temperature does to the same scores

The rows are unchanged; only the scale before the softmax moves. At 1.0 every column
gets some probability mass, at 0.1 the correct column takes almost all of it, and the
cross-entropy loss falls with it because it is computed on the same scaled scores.
"""

# Same scores at two temperatures: the softmax sharpens and the loss drops.
TEMPERATURES = (1.0, 0.1)
N_CLASSES = 3

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, temperature in zip(axes, TEMPERATURES):
    scaled = scores / temperature
    probabilities = scaled.softmax(dim=1)
    targets = torch.arange(N_CLASSES)
    loss = F.cross_entropy(scaled, targets).item()

    sns.heatmap(probabilities.numpy(), annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Blues", ax=ax, cbar=False)
    ax.set(xlabel="column", ylabel="query row", title=f"temperature {temperature}: loss {loss:.3f}")

fig.suptitle("Same scores, sharper softmax, smaller loss", fontsize=10)
fig.tight_layout()
plt.show()
