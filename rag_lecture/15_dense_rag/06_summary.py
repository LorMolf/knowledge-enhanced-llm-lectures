r"""
### The label-free summary

None of these counts needs a relevance label: how many supplied papers the answer names,
how many of those are labelled, and whether it said the evidence was insufficient. They
describe behaviour, not quality. One raw answer is printed so the counting can be
checked against the text that produced it.
"""

# Average the counts per condition, split by whether retrieval found a labelled paper.
counts = rag_frame.groupby(["condition", "top-3 holds a labelled paper"])[
    ["cited supplied papers", "cited labelled papers", "said insufficient"]
].mean()
display(counts.round(2))
print("one answer:", rag_frame.loc[0, "condition"], "->", rag_frame.loc[0, "answer"][:300])
