r"""
### Agreement is different from relevance

Two retrievers can return the same ten papers and still be wrong, or different
papers and both be right. [TrecTools](https://github.com/joaopalotti/trectools)
counts how many of the top ten papers two runs share, averaged over the
questions; the diagonal is a run against itself, so it is always ten. It uses no
relevance labels. Which pair shares the most, and is that pair also the one with
the best scores above? The next cell answers the other half: which questions
each run actually solves.
"""

from trectools import TrecRun

# Convert each ranking into a TrecRun, one row per (query, doc, rank, score)
trec_runs = {}
for name in [name for name in ("BM25", "BGE", "BM25+BGE RRF", "HippoRAG2") if name in all_rankings]:
    rows = [
        {
            "query": qid,
            "q0": "Q0",
            "docid": hit["corpus_id"],
            "rank": rank,
            "score": hit["score"],
            "system": name,
        }
        for qid, hits in all_rankings[name].items()
        for rank, hit in enumerate(hits, 1)
    ]
    run = TrecRun()
    run.load_run_from_dataframe(pd.DataFrame(rows))
    trec_runs[name] = run

# Pairwise top-K overlap: diagonal is a run against itself, so it equals MAX_K
trec_names = list(trec_runs)
overlap = pd.DataFrame(
    [
        [trec_runs[one].check_run_coverage(trec_runs[other], topX=MAX_K) for other in trec_names]
        for one in trec_names
    ],
    index=trec_names,
    columns=trec_names,
)

# Show the agreement matrix; off-diagonal cells are the interesting ones
display(overlap.round(1))
