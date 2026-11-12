r"""
### Inspect the ranked scores

The order matters, but so does the score pattern. The first question's top ten, with the
papers the qrels judge relevant marked in green: put the rank on the axis, because an
eight-digit id tells a reader nothing, and print the excerpts instead. A small score gap is
not a probability, and an unjudged paper is not necessarily irrelevant — which of these
would you open first to understand the error?
"""

# 1. the ids and scores of the first question's top ten, in rank order
ranked_ids = [hit["corpus_id"] for hit in first_hits]
ranked_scores = [hit["score"] for hit in first_hits]
relevant_flags = [doc_id in relevant_docs[query_ids[0]] for doc_id in ranked_ids]

# 2. green for judged relevant, grey otherwise
colors = ["tab:green" if flag else "tab:gray" for flag in relevant_flags]

# 3. lollipop by rank: the x axis is the information, the id is not
fig, ax = plt.subplots(figsize=(7.4, 3.2), constrained_layout=True)
ranks = range(1, len(first_hits) + 1)
ax.vlines(ranks, min(ranked_scores), ranked_scores, color=colors, alpha=0.45)
ax.scatter(ranks, ranked_scores, color=colors, s=45, zorder=3)
ax.set_ylim(min(ranked_scores) - 0.01, max(ranked_scores) + 0.01)
ax.set(xticks=list(ranks), xlabel="rank", ylabel="cosine similarity",
       title="the first question's top ten: green = judged relevant")
plt.show()

# 4. the first relevant hit, and the gap between rank 1 and it
first_relevant = next((rank for rank, flag in zip(ranks, relevant_flags) if flag), None)
print(f"first judged-relevant paper at rank {first_relevant} of {len(ranked_ids)}"
      if first_relevant else "no judged-relevant paper in the top ten")
if first_relevant:
    print(f"score gap rank1 - rank{first_relevant}: {ranked_scores[0] - ranked_scores[first_relevant - 1]:.4f}")

# 5. the text of the top three, so the miss can be judged rather than trusted
for rank, doc_id in enumerate(ranked_ids[:3], 1):
    print(f"{rank}. {'relevant' if relevant_flags[rank - 1] else 'unjudged'} | {corpus[doc_id][:150].replace(chr(10), ' ')}")
