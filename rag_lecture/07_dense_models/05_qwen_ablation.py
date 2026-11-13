r"""
### Compare Qwen query prompts

Only the query prompt changes; the same unprompted document vectors are used
for both runs. Which documents move for the first question? Compare the
aggregate metrics too, remembering that unlabelled papers may still be useful.
"""

# encode the same queries twice: once with the instruction prompt, once with an empty prompt
qwen_prompted = qwen_model.encode(
    query_texts, prompt=QWEN_QUERY_PROMPT, convert_to_tensor=True, normalize_embeddings=True
).cpu()
qwen_empty = qwen_model.encode(
    query_texts, prompt="", convert_to_tensor=True, normalize_embeddings=True
).cpu()

# rank the shared corpus vectors against each query set
qwen_prompted_rankings = rank_dense(qwen_prompted, qwen_corpus_embeddings, query_ids)
qwen_empty_rankings = rank_dense(qwen_empty, qwen_corpus_embeddings, query_ids)

# show aggregate metrics side by side, plus the top-3 hits for the first question
display(
    {
        "prompted": evaluate_run("Qwen instruction", qwen_prompted_rankings),
        "empty": evaluate_run("Qwen empty prompt", qwen_empty_rankings),
        "first_question": query_texts[0],
        "prompted_hits": [
            {**hit, "excerpt": corpus[hit["corpus_id"]][:400]}
            for hit in qwen_prompted_rankings[query_ids[0]][:3]
        ],
        "empty_hits": [
            {**hit, "excerpt": corpus[hit["corpus_id"]][:400]}
            for hit in qwen_empty_rankings[query_ids[0]][:3]
        ],
    }
)

# free the previous model and its tensors before the next section
del qwen_model, qwen_corpus_embeddings, qwen_prompted, qwen_empty
