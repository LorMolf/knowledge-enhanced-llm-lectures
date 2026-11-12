r"""
### A second retrieval step

Ask for one alternative search phrase after the first evidence, then search
again. Compare it to the first actual tool-chosen query and its papers, using
the same k. This one-question illustration is qualitative, not a measured
rewrite improvement across the test set. Did the second query find a different
paper, and does its text support the answer?
"""

# 1. extend the tool conversation with the first answer and a rewrite request
rewrite_messages = tool_messages + [
    first_tool_answer,
    {
        "role": "user",
        "content": "Write one alternative concise biomedical search query to find complementary evidence. Return only the query.",
    },
]

# 2. ask the model for the alternative query and guard against empty output
rewrite_content = chat(rewrite_messages).get("content")
if not isinstance(rewrite_content, str) or not rewrite_content.strip():
    raise ValueError("Model did not return a nonempty text rewrite")
rewrite = rewrite_content.strip()

# 3. without a local model no tool call happened: fall back to the user's own question
if not first_tool_searches:
    print("no tool call happened (no local model): using the question itself as the first query")
    first_tool_searches = [{"query": example_question, "k": 5,
                            "hits": search_papers(example_question, k=5)}]

# 4. reuse the first tool search's k so the two retrievals are comparable
first_tool_search = first_tool_searches[0]
second_retrieval = search_papers(rewrite, k=first_tool_search["k"])

# 4. show first query, rewritten query, and its hits side by side
display(
    {
        "question": example_question,
        "first_tool_search": first_tool_search,
        "rewrite": rewrite,
        "rewritten_hits": second_retrieval,
    }
)
