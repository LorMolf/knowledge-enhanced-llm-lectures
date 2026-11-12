r"""
### Dot product, cosine and L2 on the same three vectors

These **illustrative** 2D arrows are not model embeddings. Dot product depends
on magnitude and direction; cosine $q^\top d/(\|q\|\|d\|)$ compares direction;
Euclidean L2 distance measures separation, where smaller is nearer. Let's
compare the same query with a long tilted vector and a short aligned vector.
Do all three measures pick the same candidate?
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F

# 1. define the toy query and two candidate vectors
# (kept here so the next cell can score and draw them)
toy_query = torch.tensor([1.0, 0.0])
toy_docs = torch.tensor([[2.0, 1.0], [0.8, 0.0]])

# 2. score each candidate with dot, cosine, and L2
toy_table = pd.DataFrame(
    {
        "candidate": ["long tilted", "short aligned"],
        "dot": (toy_docs @ toy_query).numpy(),
        "cosine": F.cosine_similarity(toy_docs, toy_query.unsqueeze(0)).numpy(),
        "L2 distance": torch.linalg.vector_norm(toy_docs - toy_query, dim=1).numpy(),
    }
)
display(toy_table)
