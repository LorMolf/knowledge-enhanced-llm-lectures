r"""
<a name="section-04"></a>

# 04. Explore actual embeddings with Atlas

### Turn a matrix into something a widget can draw

The widget needs a table, not a tensor: two coordinates per point, the text, and the
neighbours it should offer when a point is clicked. PCA is fast and linear, so it is the
default; UMAP is slower but keeps local structure, which is what the corpus view needs.

"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def atlas_frame(texts, vectors, kinds, ids, method="pca"):
    """Coordinates plus nearest neighbours, one row per text."""
    vectors = np.asarray(vectors.detach().float().cpu() if torch.is_tensor(vectors) else vectors)
    assert len(texts) == len(vectors) == len(kinds) == len(ids) and len(texts) >= 3

    # 1. the 2-D layout: PCA by default because it is instant
    if method == "umap":
        from umap import UMAP
        xy = UMAP(n_neighbors=min(15, len(texts) - 1), min_dist=0.1, metric="cosine",
                  random_state=SEED).fit_transform(vectors)
    else:
        xy = PCA(n_components=2, random_state=SEED).fit_transform(vectors)

    # 2. neighbours in the original space, not in the projection
    nn = NearestNeighbors(n_neighbors=min(6, len(texts)), metric="cosine").fit(vectors)
    distances, indices = nn.kneighbors(vectors)
    neighbors = [
        {"ids": [int(j) for j in ix if j != i], "distances": [float(d) for j, d in zip(ix, ds) if j != i]}
        for i, (ix, ds) in enumerate(zip(indices, distances))
    ]
    return pd.DataFrame({"text": texts, "kind": kinds, "source_id": ids,
                         "x": xy[:, 0], "y": xy[:, 1], "neighbors": neighbors})


print("atlas_frame defined: PCA by default, UMAP on demand")
