r"""
### Three adapters, three runs

Each task is a separate run in the ledger, so the comparison table at the end can answer
which objective suits retrieval. The clustering adapter also gets its atlas, because that
is the space where its structure is supposed to be visible.

"""

# Loop over the three task adapters; each one re-encodes both sides from scratch.
for jina_task in ("text-matching", "clustering", "classification"):
    jina_docs, jina_queries = encode_task(jina_model, jina_task)

    # the clustering space is the one worth looking at
    if jina_task == "clustering":
        atlas_cluster_vectors = torch.cat((jina_queries[atlas_query_indices], jina_docs[atlas_doc_indices]))
        display(show_atlas(atlas_texts, atlas_cluster_vectors, atlas_kinds, atlas_ids))

    # rank, score into the ledger, and show the first question's top three
    jina_hits = rank_dense(jina_queries, jina_docs, query_ids)
    display({
        "task": jina_task,
        "metrics": evaluate_run(f"Jina {jina_task}", jina_hits),
        "question": query_texts[0],
        "first_question_hits": [{**hit, "excerpt": corpus[hit["corpus_id"]][:400]}
                                for hit in jina_hits[query_ids[0]][:3]],
    })

del jina_docs, jina_queries, jina_hits
