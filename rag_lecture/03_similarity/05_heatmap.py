r"""
### Where the labelled paper actually lands

Five questions, one dot each: the rank the labelled paper reaches among all 2,000 papers
under the CLS encoder, with the score gap to the paper that came first. A dot at 1 is a
working retriever; a dot at 4 with a gap of 0.01 is a near miss; a dot past 100 says the
encoder never saw the connection. The worst case is then printed in words, both papers,
so the miss can be judged rather than trusted.
"""

# 1. the labelled paper's true rank in the full corpus, not just the top ten
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

# 2. one dot per question on a log rank axis: 1 is perfect, 1000 is hopeless
fig, ax = plt.subplots(figsize=(7.2, 2.8), constrained_layout=True)
y = np.arange(len(table))
ax.hlines(y, 1, table["labelled rank"], color="#C9CFD8", linewidth=2)
ax.scatter(table["labelled rank"], y, s=60, color="#9D2235", zorder=3, label="labelled paper")
ax.set_yticks(y, [f"Q{i}" for i in range(1, len(table) + 1)], fontsize=8)
ax.set(xlabel=f"rank of the labelled paper among {len(corpus_ids):,} papers", xscale="log",
       xticks=[1, 10, 100, 1000], xlim=(0.9, len(corpus_ids) * 1.3))
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("how deep the first right paper sits, five questions", fontsize=9, loc="left")
plt.show()

# 3. the worst row in words: the question, the labelled paper, and what beat it
worst = table["labelled rank"].idxmax()
print(f"Q{list(table.index).index(worst) + 1} ({worst}):", queries[worst][:110])
print("labelled paper:", corpus[sorted(relevant_docs[worst])[0]][:200].replace("\n", " "))
rank1_id = table.loc[worst, "rank-1 paper"]
print("what came first:", corpus[rank1_id][:200].replace("\n", " "))
