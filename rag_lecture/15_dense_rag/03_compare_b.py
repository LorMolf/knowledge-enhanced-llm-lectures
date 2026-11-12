r"""
### Comparing the three conditions

The no-context answer is generated here from the same question, so the comparison does not
reuse the answer section 14 happened to print; it still needs that server to be up.

Now we answer with retrieved evidence (what a deployed system would see) and with oracle evidence (an upper-bound diagnostic, not deployable). The table shows all three conditions side by side for qualitative comparison. Relevance labels are incomplete, so neither a retrieved nor an oracle paper guarantees a correct answer.
"""

# 1. answer with retrieved evidence (what a deployed system would see)
retrieved_answer = chat(
    [
        rag_system,
        {
            "role": "user",
            "content": f"Question: {example_question}\nEvidence: {json.dumps(retrieved)}",
        },
    ]
)

# 2. answer with oracle evidence (upper-bound diagnostic, not deployable)
gold_answer = chat(
    [
        rag_system,
        {
            "role": "user",
            "content": f"Question: {example_question}\nEvidence: {json.dumps(gold_evidence)}",
        },
    ]
)

# 3. the same question without evidence, generated here so this cell stands alone
no_context = chat([{"role": "user", "content": f"Question: {example_question}"}])

# 4. show the three conditions side by side for qualitative comparison
display(
    pd.DataFrame(
        [
            {"condition": "no context", "answer": no_context.get("content")},
            {"condition": "BGE top-3", "answer": retrieved_answer.get("content")},
            {"condition": "labelled papers", "answer": gold_answer.get("content")},
        ]
    )
)
