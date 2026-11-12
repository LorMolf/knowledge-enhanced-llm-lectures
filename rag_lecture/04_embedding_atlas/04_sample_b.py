r"""
The enriched sample keeps the neighbour tables exact. The whole corpus gets one map of its own below, from UMAP instead of PCA: 2,000 documents and no questions, about a minute on a CPU the first time. Zoom into a dense region and read its documents: are they about one topic, or from one journal? A 40-point sample can answer neither question, and a PCA view flattens clusters that UMAP keeps apart.
"""

# 5. nearest sampled documents per question in CLS space
cls_doc_nn = NearestNeighbors(n_neighbors=min(5, len(atlas_doc_indices)), metric="cosine")
cls_doc_nn.fit(atlas_cls_doc_vectors.numpy())
cls_indices = cls_doc_nn.kneighbors(atlas_cls_query_vectors.numpy(), return_distance=False)
atlas_cls_neighbors = [[corpus_ids[atlas_doc_indices[j]] for j in row] for row in cls_indices]

# 6. compare neighbours against the labelled relevant IDs
display(
    pd.DataFrame(
        {
            "question_id": [query_ids[i] for i in atlas_query_indices],
            "CLS sampled document neighbors": atlas_cls_neighbors,
            "labelled IDs": [sorted(relevant_docs[query_ids[i]]) for i in atlas_query_indices],
        }
    )
)

# 7. whole-corpus UMAP map, no questions, to read cluster topics
atlas_corpus_widget = show_atlas(
    corpus_texts,
    naive_corpus_embeddings,
    ["document"] * len(corpus_ids),
    corpus_ids,
    method="umap",
)
display(atlas_corpus_widget)
