r"""
### Where fusion moved each paper

One question, three stages: BM25's list, the fused list, BGE's list. Each line is a
paper, its height is the rank it holds at that stage, and green means the dataset
labels it relevant for this question. A line that climbs from left to right gained
from fusion; a paper missing from a stage sits at rank 11. Which papers did BM25 and
BGE disagree about, and did fusion pick the right one?
"""

import matplotlib.pyplot as plt

# the three stages, and every paper that appears in any of them
stage_names = ["BM25", "RRF", "BGE"]
stage_rankings = {"BM25": bm25_rankings, "RRF": hybrid_rankings, "BGE": all_rankings["BGE"]}

# invert each stage's list into a corpus_id -> rank lookup
ranks = {}
for name in stage_names:
    ranks[name] = {}
    for rank, hit in enumerate(stage_rankings[name][example_qid], 1):
        ranks[name][hit["corpus_id"]] = rank

# collect the union of papers across stages, capped so the figure stays readable
bump_ids = []
for name in stage_names:
    for hit in stage_rankings[name][example_qid]:
        if hit["corpus_id"] not in bump_ids:
            bump_ids.append(hit["corpus_id"])
bump_ids = bump_ids[:12]

# one line per paper, drawn top-down by rank; green = judged relevant
fig, ax = plt.subplots(figsize=(6, 0.35 * len(bump_ids) + 2))
for doc_id in bump_ids:
    trail = [ranks[name].get(doc_id, MAX_K + 1) for name in stage_names]
    labelled = doc_id in relevant_docs[example_qid]
    ax.plot(range(len(stage_names)), trail, marker="o", lw=1.5,
            color="tab:green" if labelled else "lightgray")
    ax.annotate(doc_id, (0, trail[0]), xytext=(-6, 0), textcoords="offset points",
                ha="right", fontsize=7)

# label axes so the reader can read rank off the y-axis
ax.set_xticks(range(len(stage_names)), stage_names)
ax.set_yticks(range(1, MAX_K + 2))
ax.invert_yaxis()
ax.set(xlabel="stage", ylabel=f"rank (missing = {MAX_K + 1})",
       title="One question through fusion: green = judged relevant")
plt.tight_layout()
plt.show()
