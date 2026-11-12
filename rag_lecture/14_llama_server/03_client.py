r"""
### The client, and the thinking switch

`llama-server` exposes the same `/v1/chat/completions` route as OpenAI, so the client
is a POST. The one extra key is `chat_template_kwargs`: it is passed into the model's
own chat template, so `{"enable_thinking": false}` turns reasoning off for *that
request* while the server keeps running — the per-request half of the switch, next to
the `--reasoning` flag that decided the default at launch. `chat_full()` keeps the
whole response, because the thinking demo needs the token counts the message drops.

Gemma 4's template turns that flag into its own `<|think|>` trigger, so the same
three lines work for any model whose template declares `enable_thinking`.

> The last two prints ask the server what it believes it is serving; they should match
> `LLM_MODEL` and the context size we launched with.
"""

import requests

# 1. LLM_URL and LLM_MODEL come from the cell that started the server.
def chat(messages, tools=None, reasoning=False):
    """One chat completion; reasoning=True makes the model think before it answers."""
    # One guard for every caller: if section 14 could not start the server, answer with a
    # stub so sections 15-17 still run and show their tables instead of stopping the lecture.
    if not SERVER_READY:
        return {"role": "assistant", "content": "[stub: no local model]"}

    # Answer-only by default: section 15 makes 96 calls, and a trace per call triples
    # the budget for no gain a reader can see. Section 14 shows the thinking answer.
    # 2. the OpenAI-shaped payload, with the template switch spelled out
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024 if reasoning else 256,
        "chat_template_kwargs": {"enable_thinking": reasoning},
    }
    if tools is not None:
        payload["tools"] = tools

    # 3. one POST, then surface HTTP errors instead of returning junk
    response = requests.post(f"{LLM_URL}/chat/completions", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


def chat_full(messages, reasoning=True):
    """The same call, returning usage too: thinking costs tokens you never see."""
    if not SERVER_READY:
        return {"role": "assistant", "content": "[stub: no local model]"}, {"total_tokens": 0}

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024 if reasoning else 256,
        "chat_template_kwargs": {"enable_thinking": reasoning},
    }
    response = requests.post(f"{LLM_URL}/chat/completions", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


# 4. what the server believes it is serving
served = requests.get("http://127.0.0.1:8080/props", timeout=10).json()
print("served model:", served.get("model_alias"), "| context:", served.get("default_generation_settings", {}).get("n_ctx"))
print("reply fields: content, reasoning_content, usage")
