r"""
### A fixed sample for two models

We include one known positive per sampled question, then fill with other corpus documents. This deliberately enriched visual sample is not the full retrieval evaluation. Select points in the widget; `atlas_cls_widget.selection()` returns selected rows, including `source_id`. Below are each question's nearest sampled **documents** in the original CLS space. Which IDs are labelled relevant? Compare these same IDs with the BGE view later.
"""

# pick the questions and one known positive document per question
atlas_query_indices = list(range(min(8, len(query_ids))))
corpus_positions = {doc_id: i for i, doc_id in enumerate(corpus_ids)}
atlas_positive_indices = {
    corpus_positions[sorted(relevant_docs[query_ids[i]])[0]] for i in atlas_query_indices
}

# fill the sample up to 32 documents with other corpus rows
atlas_doc_indices = sorted(atlas_positive_indices)
atlas_doc_indices += [i for i in range(len(corpus_ids)) if i not in atlas_positive_indices][
    : max(0, 32 - len(atlas_doc_indices))
]

# gather ids, texts and kinds for the sampled rows
atlas_ids = [query_ids[i] for i in atlas_query_indices] + [corpus_ids[i] for i in atlas_doc_indices]
atlas_texts = [query_texts[i] for i in atlas_query_indices] + [
    corpus_texts[i] for i in atlas_doc_indices
]
atlas_kinds = ["question"] * len(atlas_query_indices) + ["document"] * len(atlas_doc_indices)

# slice the CLS embeddings for the sample and draw the atlas
atlas_cls_query_vectors = naive_query_embeddings[atlas_query_indices]
atlas_cls_doc_vectors = naive_corpus_embeddings[atlas_doc_indices]
atlas_vectors = torch.cat((atlas_cls_query_vectors, atlas_cls_doc_vectors))
atlas_cls_widget = show_atlas(atlas_texts, atlas_vectors, atlas_kinds, atlas_ids)
display(atlas_cls_widget)
