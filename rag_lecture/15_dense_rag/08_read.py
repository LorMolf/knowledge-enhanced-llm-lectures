r"""
### Read the two answers the proxy liked most, and least

A number cannot tell you whether an answer is right, so look at the extremes. If
the highest-scoring answers are on topic and well cited, the proxy is doing part of
its job; if one of them is confidently wrong, that is the lesson: cosine to a
reference is not correctness. Then check a low one: is it wrong, or just worded
differently from the reference?
"""

# pick the two highest and the two lowest by the proxy score only
for label, selection in (
    ("highest", rag_frame.nlargest(2, "similarity to reference (proxy)")),
    ("lowest", rag_frame.nsmallest(2, "similarity to reference (proxy)")),
):
    print(f"--- {label} two answers ---")
    # show condition, rounded score, then the answer text itself
    for _, row in selection.iterrows():
        score = round(row["similarity to reference (proxy)"], 3)
        print(row["condition"], "|", score, "|", row["answer"][:400])
