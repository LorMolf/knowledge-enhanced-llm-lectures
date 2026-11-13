r"""
<a name="section-15"></a>

# Dense retrieval-augmented generation

![A question travels through retrieval and generation](assets/rag-pipeline.png)

*Figure: naive, advanced and modular RAG, from [Gao et al., 2023](https://arxiv.org/abs/2312.10997), Fig. 3.*

The BGE baseline from section 07 maps the question and papers into a shared
space. We reuse its **unchanged** document vectors and published query instruction.
Retrieved paper text is evidence, never an instruction to the assistant.

**Question:** what is the highest-scoring paper for our question?

For deterministic ties we score the whole small teaching corpus, then order by
score and original paper ID; this full scan is not a large-scale index design.
"""

import pandas as pd
from sentence_transformers import util

# Wide columns so the retrieved paper text is readable in the table.
pd.set_option("display.max_colwidth", 400)


def search_papers(query, k=5):
    # guard: reject empty queries and out-of-range k before encoding
    if not isinstance(query, str) or not query.strip() or type(k) is not int or not 1 <= k <= 10:
        raise ValueError("query must be nonempty and k an integer from 1 to 10")

    # encode the question with the published BGE query instruction
    vector = bge_model.encode([query], prompt=BGE_QUERY_PROMPT, convert_to_tensor=True)

    # score every paper (full scan) so ties are resolved deterministically
    hits = util.semantic_search(vector.cpu(), bge_corpus_embeddings.cpu(), top_k=len(corpus_ids))[0]

    # order by score desc, then by original paper ID for stable ties
    hits.sort(key=lambda h: (-float(h["score"]), corpus_ids[h["corpus_id"]]))

    # return the top-k as plain dicts: id, text, score
    return [
        {
            "id": corpus_ids[h["corpus_id"]],
            "text": corpus[corpus_ids[h["corpus_id"]]],
            "score": float(h["score"]),
        }
        for h in hits[:k]
    ]


# run the search on the lecture question and show the ranked evidence
retrieved = search_papers(example_question, k=3)
print("Question:", example_question)
display(pd.DataFrame(retrieved))
