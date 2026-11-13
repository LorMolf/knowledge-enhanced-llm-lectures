r"""
### The same question, thinking on and off

One request key changes how much work the model does before it answers: with
`enable_thinking` on, Gemma 4 emits its thought channel before the answer and
llama-server returns that text in `reasoning_content`, separate from `content`.
Compare the four numbers: wall clock, completion tokens, length of the trace and
length of the answer. The baseline the rest of the section compares against is the
answer without thinking: retrieval supplies the evidence, not a longer deliberation.
"""

import time

# one question from the frozen set, asked twice with identical sampling
example_qid = query_ids[0]
example_question = queries[example_qid]


def timed(question, reasoning):
    """One call, plus how long it took; the payload carries usage and the trace."""
    started = time.monotonic()
    payload = chat_full([{"role": "user", "content": question}], reasoning=reasoning)
    return payload, time.monotonic() - started


# the switch, per request: same server, same weights, different template kwarg
thinking, seconds_on = timed(example_question, True)
direct, seconds_off = timed(example_question, False)

for label, payload, seconds in (("thinking", thinking, seconds_on), ("direct", direct, seconds_off)):
    message = payload["choices"][0]["message"]
    thoughts = message.get("reasoning_content") or ""
    print(
        f"{label}: {seconds:.1f}s | completion tokens {payload['usage']['completion_tokens']}"
        f" | trace {len(thoughts)} chars | answer {len(message['content'])} chars"
    )

# what the two answers actually say, and the trace that produced the first one
print("\nQuestion:", example_question)
print("Answer with thinking off:", direct["choices"][0]["message"]["content"][:400])
print("Trace, first 300 chars:", (thinking["choices"][0]["message"].get("reasoning_content") or "")[:300])

no_context = direct["choices"][0]["message"]
