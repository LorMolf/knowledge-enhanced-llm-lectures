r"""
### Look at the same joint sample again

Section 04 fixed the same ordered question and document IDs used here. The
visual sample is enriched with labelled-positive documents and is not the full
corpus. Each model gets its own PCA, so compare the original-space neighbours,
not axis positions. Which sampled neighbours and labelled IDs are shared?
"""

import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# gather the same joint sample of questions and documents for BGE
atlas_bge_q = bge_query_embeddings[atlas_query_indices]
atlas_bge_d = bge_corpus_embeddings[atlas_doc_indices]
atlas_bge_vectors = torch.cat((atlas_bge_q, atlas_bge_d))

# draw the BGE PCA view of the sample
display(show_atlas(atlas_texts, atlas_bge_vectors, atlas_kinds, atlas_ids))

# fit a cosine kNN on the BGE document vectors
atlas_nn = NearestNeighbors(metric="cosine", n_neighbors=len(atlas_cls_neighbors[0]))
atlas_nn.fit(atlas_bge_d.numpy())

# query the same questions and map row indices back to corpus IDs
atlas_positions = atlas_nn.kneighbors(atlas_bge_q.numpy(), return_distance=False)
atlas_bge_neighbors = [[corpus_ids[atlas_doc_indices[i]] for i in row] for row in atlas_positions]

# compare CLS vs BGE neighbours and labelled positives per question
atlas_comparison = []
for i, qi in enumerate(atlas_query_indices):
    qid = query_ids[qi]
    cls_ids, bge_ids = atlas_cls_neighbors[i], atlas_bge_neighbors[i]
    atlas_comparison.append(
        {
            "question_id": qid,
            "question": query_texts[qi],
            "CLS IDs": cls_ids,
            "BGE IDs": bge_ids,
            "shared IDs": sorted(set(cls_ids) & set(bge_ids)),
            "labelled IDs": sorted(relevant_docs[qid]),
            "CLS labelled": sorted(set(cls_ids) & relevant_docs[qid]),
            "BGE labelled": sorted(set(bge_ids) & relevant_docs[qid]),
        }
    )

# show the side-by-side comparison table
display(pd.DataFrame(atlas_comparison))

# free the accelerator: keep the same baseline weights on CPU
bge_model.to("cpu")
