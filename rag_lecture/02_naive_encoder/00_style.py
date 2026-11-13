r"""
### One figure style for the whole lecture

Papers in retrieval share four habits: a muted palette, no chartjunk, labels on the lines
instead of a legend box, and an uncertainty band wherever a mean is drawn. Setting them
once in `rcParams` means every figure after this cell inherits the look, and `bootstrap_ci`
gives the band those papers draw over questions rather than over seeds.
"""

import matplotlib.pyplot as plt
import numpy as np

# Four colours and one set of rcParams: every later figure inherits both.
INK, ACCENT, MUTED = "#334155", "#B45309", "#94A3B8"
OURS, BASELINE = "#0F766E", "#64748B"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "figure.figsize": (6.4, 3.2),
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.5, "lines.markersize": 3.5,
})


def panel(ax, xlabel="", ylabel="", note=""):
    """Labels plus a left-aligned caption line; the spines are already trimmed by rcParams."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if note:
        ax.set_title(note, loc="left", color=INK, pad=8)
    return ax


def bootstrap_ci(values, n_boot=2000, seed=None):
    """95% percentile bootstrap over the questions: the band an IR paper draws around a mean."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    return draws.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


print("style set | font", plt.rcParams["font.size"], "pt | ours", OURS, "| baseline", BASELINE)
