r"""
### Where the labelled paper actually lands

Five questions, one dot each: the rank the labelled paper reaches among all 2,000 papers
under the CLS encoder, with the score gap to the paper that came first. A dot at 1 is a
working retriever; a dot at 4 with a gap of 0.01 is a near miss; a dot past 100 says the
encoder never saw the connection. The worst case is then printed in words, both papers,
so the miss can be judged rather than trusted.
"""

# the labelled paper's true rank in the full corpus, not just the top ten
rows = []
normalised_corpus = F.normalize(naive_corpus_embeddings, dim=1)
for position, qid in enumerate(query_ids[:5]):
    scores = (F.normalize(naive_query_embeddings[position:position + 1], dim=1)
              @ normalised_corpus.T).squeeze(0)
    order = torch.argsort(scores, descending=True)
    ranked_ids = [corpus_ids[i] for i in order.tolist()]
    labelled = sorted(relevant_docs[qid])[0]
    rows.append({
        "question id": qid,
        "labelled rank": ranked_ids.index(labelled) + 1,
        "score of rank 1": float(scores[order[0]]),
        "score of labelled": float(scores[order[ranked_ids.index(labelled)]]),
        "rank-1 paper": ranked_ids[0],
    })

table = pd.DataFrame(rows).set_index("question id")
table["gap to rank 1"] = (table["score of rank 1"] - table["score of labelled"]).round(3)
display(table)

# one dot per question on a log rank axis: 1 is perfect, 1000 is hopeless
fig, ax = plt.subplots(constrained_layout=True)
ranks = table["labelled rank"].to_numpy(float)
rng = np.random.default_rng(SEED)
ax.scatter(ranks, rng.uniform(-0.16, 0.16, len(ranks)), s=34, color=OURS,
           alpha=0.9, edgecolor="white", linewidth=0.6, zorder=3)
ax.axvline(np.median(ranks), color=ACCENT, linestyle=":", linewidth=1, zorder=2)
ax.annotate(f"median {int(np.median(ranks)):,}", (np.median(ranks), 0.26),
            textcoords="offset points", xytext=(4, 0), fontsize=7, color=ACCENT)
ax.set(xscale="log", xlim=(0.85, len(corpus_ids) * 1.4), ylim=(-0.34, 0.34),
       xticks=[1, 10, 100, 1000], yticks=[])
panel(ax, f"rank of the labelled paper among {len(corpus_ids):,}", "",
      "where the first right paper sits, one dot per question\nlog scale; the dotted line is the median")
plt.show()

# the worst row in words: the question, the labelled paper, and what beat it
worst = table["labelled rank"].idxmax()
print(f"Q{list(table.index).index(worst) + 1} ({worst}):", queries[worst][:110])
print("labelled paper:", corpus[sorted(relevant_docs[worst])[0]][:200].replace("\n", " "))
rank1_id = table.loc[worst, "rank-1 paper"]
print("what came first:", corpus[rank1_id][:200].replace("\n", " "))
