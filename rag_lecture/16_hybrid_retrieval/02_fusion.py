r"""
### Reciprocal rank fusion

RRF combines **ranks**, not scores: $\mathrm{RRF}(d)=\sum_r 1/(60+\mathrm{rank}_r(d))$.
That is what lets a BM25 score and a cosine live in the same list, and it rewards a
paper that both retrievers like without pretending the two numbers are comparable.
We fuse the existing depth-10 lists, so a paper that neither retriever returned
cannot come back: this is fusion of two candidate lists, not full-corpus rescoring.
[ranx fusion](https://amenra.github.io/ranx/fusion/). Which paper ends first?
"""

from ranx import Run, fuse


def as_run(rankings, name):
    """A ranx Run from our hits; negative ranks keep the existing order under ties."""
    return Run(
        {
            qid: {hit["corpus_id"]: -float(i) for i, hit in enumerate(hits)}
            for qid, hits in rankings.items()
        },
        name=name,
    )


# wrap the two candidate lists as ranx runs
bm25_run = as_run(bm25_rankings, "BM25")
bge_run = as_run(all_rankings["BGE"], "BGE")

# fuse by rank, k = 60 as in the formula
fused = fuse(runs=[bm25_run, bge_run], method="rrf", params={"k": 60})

# back to the lecture's ranking shape: score descending, then ID ascending
hybrid_rankings = {}
for qid in queries:
    ordered = sorted(fused.run[qid].items(), key=lambda item: (-item[1], item[0]))[:MAX_K]
    hybrid_rankings[qid] = [
        {"corpus_id": doc_id, "score": float(score)} for doc_id, score in ordered
    ]

# compare the top paper from each retriever against the fused winner
print(
    "question",
    example_qid,
    "| BM25 first:",
    bm25_rankings[example_qid][0]["corpus_id"],
    "| RRF first:",
    hybrid_rankings[example_qid][0]["corpus_id"],
    "| BGE first:",
    all_rankings["BGE"][example_qid][0]["corpus_id"],
)
