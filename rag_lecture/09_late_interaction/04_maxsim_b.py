r"""
The hand-rolled sum of row maxima should match the native ColBERT scorer for the same pair. Comparing the two numbers confirms that the matrix view is the same MaxSim operation the ranking uses.
"""

# compare the hand-rolled MaxSim sum with the native scorer
print("Illustrative sum of row maxima:", similarities.max(dim=1).values.sum().item())
print(
    "Native score:",
    colbert_model.similarity(
        [colbert_queries[0]], [colbert_docs[inspect_doc_index]], device=DEVICE
    ).cpu(),
)
