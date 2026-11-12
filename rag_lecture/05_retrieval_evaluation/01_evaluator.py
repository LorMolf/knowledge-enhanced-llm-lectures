r"""
<a name="section-05"></a>

# 05. Evaluate retrieval

### The evaluator, configured once

All six metrics come from one object: accuracy@k (was a labelled paper in the first
k), precision@k, recall@k, MRR, nDCG and MAP. Two copies are built because the
lecture uses two question sets: the frozen test questions and the validation
questions that training may look at. Unjudged documents count as non-relevant, so a
missing label can only lower a score.

The next cell defines the one function that may write to the shared ledger; this
cell only says which questions and documents it will score.
"""

from sentence_transformers.evaluation import InformationRetrievalEvaluator

# 1. one configuration, used for both question sets
METRIC_KWARGS = dict(
    mrr_at_k=KS,
    ndcg_at_k=KS,
    accuracy_at_k=KS,
    precision_recall_at_k=KS,
    map_at_k=KS,
    write_csv=False,
)

# 2. test questions: never used for training or for choosing a model
evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs, **METRIC_KWARGS)

# 3. validation questions: the only signal training may use
validation_evaluator = InformationRetrievalEvaluator(
    val_queries, corpus, val_relevant_docs, **METRIC_KWARGS
)

print(
    "test questions:",
    len(evaluator.queries_ids),
    "| validation questions:",
    len(validation_evaluator.queries_ids),
    "| corpus:",
    len(corpus),
    "| cutoffs:",
    KS,
)
