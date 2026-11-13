r"""
### What fusion bought, and which paper came first

Same evaluator, three rows: BM25 alone, BGE alone, and the fused list. The numbers
say whether mixing a lexical and a dense ranking helped on these 32 questions; the
paper printed below says what the new first hit actually is. Is it labelled, and
would you have found it with only one of the two retrievers?
"""

import pandas as pd

# score the fused list with the same evaluator as every other run
rrf_row = evaluate_run("BM25+BGE RRF", hybrid_rankings)

# the three rows side by side, so the comparison is in the same cell
three_rows = pd.DataFrame(evaluation_rows).set_index("model")
three_rows = three_rows.reindex(["BM25", "BGE", "BM25+BGE RRF"]).dropna(how="all")
display(three_rows[["accuracy@1", "recall@10", "mrr@10", "ndcg@10"]])

# and the paper that fusion promoted to rank one
top_id = hybrid_rankings[example_qid][0]["corpus_id"]
print("first after fusion:", top_id, "| labelled for this question:", top_id in relevant_docs[example_qid])
print(corpus[top_id][:400])
print("RRF row:", {key: round(value, 3) for key, value in rrf_row.items() if "@10" in key})
