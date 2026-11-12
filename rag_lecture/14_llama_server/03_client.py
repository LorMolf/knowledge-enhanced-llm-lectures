r"""
### The client, with the OpenAI library

`llama_cpp.server` exposes the OpenAI API, so the client is the `openai` package pointed
at localhost — the same twelve lines a student would write against a hosted model, which
is the point: swapping the base URL is the whole difference between a laptop model and an
API. The one extra key is `chat_template_kwargs`: it reaches the model's own chat
template, so `{"enable_thinking": false}` turns reasoning off for *that request* while the
server keeps running. `chat_full()` returns the whole response, because the thinking demo
needs the token counts the message drops.

> The last print asks the server what it is serving; it should match `LLM_MODEL`.
"""

from openai import OpenAI

# 1. The client every later cell calls; LLM_URL and LLM_MODEL come from section 14's first cell.
client = OpenAI(base_url=LLM_URL, api_key="local")


def chat(messages, tools=None, reasoning=False):
    """One chat completion; reasoning=True makes the model think before it answers."""
    # One guard for every caller: if the server never came up, answer with a stub so
    # sections 15-17 still run and show their tables instead of stopping the lecture.
    if not SERVER_READY:
        return {"role": "assistant", "content": "[stub: no local model]"}

    # Answer-only by default: section 15 makes 96 calls, and a trace per call triples
    # the budget for no gain a reader can see. Section 14 shows the thinking answer.
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024 if reasoning else 256,
        tools=tools,
        extra_body={"chat_template_kwargs": {"enable_thinking": reasoning}},
    )
    return completion.choices[0].message.model_dump()


def chat_full(messages, reasoning=True):
    """The same call, returning usage too: thinking costs tokens you never see."""
    if not SERVER_READY:
        # same envelope as a real response, so the cells that read usage still work
        return {
            "choices": [{"message": {"role": "assistant", "content": "[stub: no local model]",
                                     "reasoning_content": ""}}],
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024 if reasoning else 256,
        extra_body={"chat_template_kwargs": {"enable_thinking": reasoning}},
    )
    return completion.model_dump()


# 2. What the server says it is serving, when there is one to ask.
if SERVER_READY:
    print("served models:", [entry.id for entry in client.models.list().data], "| expected:", LLM_MODEL)
else:
    print("no server to ask: section 14's server did not start, so the client returns stubs")
print("reply fields: content, tool_calls, usage")
