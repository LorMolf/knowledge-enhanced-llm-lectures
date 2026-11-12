r"""
### Both sides, under one task adapter

`task=` selects a projector trained for one objective, and `prompt_name=` picks the query
or document role inside it. The two must match: a query encoded with the document prompt
sits in the wrong place and the ranking collapses for a reason that looks like a bug.

"""

# One adapter encodes both sides; the loop over the three lives in the next cell.
def encode_task(model, jina_task):
    """Encode the corpus and the questions under one task adapter, normalised."""
    docs = model.encode(corpus_texts, task=jina_task, prompt_name="document",
                             convert_to_tensor=True, normalize_embeddings=True).cpu()
    queries_task = model.encode(query_texts, task=jina_task, prompt_name="query",
                                     convert_to_tensor=True, normalize_embeddings=True).cpu()
    return docs, queries_task


print("encode_task defined: one adapter, both sides, unit length")
