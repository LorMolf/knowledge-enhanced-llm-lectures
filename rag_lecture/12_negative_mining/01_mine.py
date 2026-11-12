r"""
<a name="section-12"></a>

# 12. Negative mining

A random negative may be easy; BM25 finds lexical neighbours, while BGE finds
semantically close documents. Harder examples can teach a useful boundary, but
partial labels mean a retrieved unjudged document might actually answer the
question. We exclude **all known training positives**, not merely the one paired
positive. We never inspect test question labels here.

We use native BM25 scoring and dense cosine similarity on the *same* frozen
corpus. Lowercase alphanumeric tokens split punctuation; this simple tokenizer
is reused for the BM25 retrieval baseline in section 16. The negative candidate
pool excludes the union of training positives, including those from dropped
questions. How much of the corpus remains eligible?
"""

import random
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import util

# 1. seed the RNG so any later sampling of negatives is reproducible
rng = random.Random(SEED)


def lexical_tokens(text):
    # 2. lowercase alphanumeric tokenizer, reused by the BM25 baseline in section 16
    return re.findall(r"[a-z0-9]+", text.lower())


# 3. build the BM25 index over the frozen corpus
lexical_index = BM25Okapi([lexical_tokens(text) for text in corpus_texts])

# 4. candidate pool = every doc that is not a known training positive
eligible = [j for j, doc_id in enumerate(corpus_ids) if doc_id not in all_training_positive_ids]
print(
    "Negative candidate pool:",
    len(eligible),
    "of",
    len(corpus_ids),
    "documents; excluded training positives:",
    len(corpus_ids) - len(eligible),
)
assert eligible, "No candidate documents remain after excluding all training positives."

# 5. encode training queries with BGE so we can score them against the corpus
# section 07 parked BGE on CPU for the Atlas comparison; mining wants it back on the GPU
bge_model = bge_model.to(DEVICE)
train_bge_vectors = bge_model.encode(
    [train_queries[qid] for qid in train_qids], prompt=BGE_QUERY_PROMPT, convert_to_tensor=True
).cpu()

# 6. dense cosine similarity between each training query and every corpus doc
dense_mining_scores = util.cos_sim(train_bge_vectors, bge_corpus_embeddings.cpu())
print("Dense mining score matrix:", tuple(dense_mining_scores.shape))
