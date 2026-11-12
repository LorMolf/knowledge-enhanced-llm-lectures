r"""
### Questions to pages, with instructions

Pages are stored with the document instruction of the model card and questions
are searched with its query instruction, both copied verbatim. `encode` returns
unit-length float32 vectors, so the ranking and evaluation calls are the ones we
used for EmbeddingGemma 2. This is the slowest cell of the lecture: a full page
can use up to 4,096 image tokens in a 28B-parameter model. The batch size of 4
leaves room for them; lower it if CUDA runs out of memory. After the metrics,
we count how often the two models put the same page first. Do the two encoders
agree where they are right, and where they are wrong?
"""

META_DOC_INSTRUCTION = "Understand the content of the provided document image."
META_QUERY_INSTRUCTION = "Find a document image that matches the given query."

if META_AVAILABLE:
    # 1. encode every page image with the document instruction
    meta_page_embeddings = meta_model.encode(
        [{"image": page} for page in mm_pages.values()],
        instruction=META_DOC_INSTRUCTION,
        batch_size=4,
    ).cpu()
    # 2. encode every question with the query instruction
    meta_query_embeddings = meta_model.encode(
        list(mm_queries.values()), instruction=META_QUERY_INSTRUCTION
    ).cpu()
    # 3. rank pages per query, same helper as EmbeddingGemma 2
    meta_rankings = rank_dense(
        meta_query_embeddings, meta_page_embeddings, mm_query_ids, mm_page_ids
    )
    display(evaluate_mm("MetaEncoder v1, card instructions", meta_rankings))
    # 4. compare top-1 picks between the two encoders
    same_first = [
        gemma_rankings[qid][0]["corpus_id"] == meta_rankings[qid][0]["corpus_id"]
        for qid in mm_query_ids
    ]
    both_right = [
        gemma_rankings[qid][0]["corpus_id"] in mm_relevant[qid]
        and meta_rankings[qid][0]["corpus_id"] in mm_relevant[qid]
        for qid in mm_query_ids
    ]
    print("same first page:", sum(same_first), "of", len(mm_query_ids))
    print("both models right at rank 1:", sum(both_right), "of", len(mm_query_ids))
else:
    print("MetaEncoder was skipped: no row to compare for this model.")
