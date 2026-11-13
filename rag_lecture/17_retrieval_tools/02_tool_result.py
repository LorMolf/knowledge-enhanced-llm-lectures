r"""
### Execute the requested search

Validate the requested name and arguments before using the allowlisted function.
Each tool result goes back under its matching call ID. Save the model's actual
query and returned papers: did it use the same words as the original question?
"""

# append the assistant's tool-call message so the transcript stays valid
tool_messages.append(requested)
first_tool_searches = []

# run each requested call, validating name and arguments first
for call in requested["tool_calls"]:
    if call["function"]["name"] != "search_papers":
        raise ValueError("Unexpected tool name")
    args = json.loads(call["function"]["arguments"])
    if not isinstance(args, dict) or set(args) - {"query", "k"} or "query" not in args:
        raise ValueError("Invalid tool arguments")

    # execute the allowlisted search and remember what was asked
    result = search_papers(**args)
    first_tool_searches.append({"query": args["query"], "k": args.get("k", 5), "results": result})

    # return the result under its matching call ID
    tool_messages.append(
        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}
    )

# ask the model to answer using the tool results
first_tool_answer = chat(tool_messages)

# show each query and its papers, then the final answer
for search in first_tool_searches:
    print("tool query:", search["query"], "| k:", search["k"])
    display(pd.DataFrame(search["results"]))
print("answer:", first_tool_answer["content"])
