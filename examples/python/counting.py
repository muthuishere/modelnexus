"""counting -- how big is this conversation, before you commit to sending it?

count_tokens applies the model's chat template and tokenizes. It creates no context,
decodes nothing and does not touch the KV cache, so it is safe between inferences. It
lives in the ABI because counting needs the model's vocabulary AND its parsed chat
template, and no binding holds either — a tokenizer bolted on in Python would be a
different tokenizer.

    MODELNEXUS_MODEL=/path/to/model.gguf python3 counting.py
"""

import os
import sys

import modelnexus

# Tools are part of the prompt too — the chat template renders them — so counting
# without them under-reports a tool-calling request.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_part",
            "description": "Find a spare part by model and component name",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "component": {"type": "string"},
                },
                "required": ["model", "component"],
            },
        },
    }
]


def main() -> int:
    path = os.environ.get("MODELNEXUS_MODEL")
    if not path:
        print("MODELNEXUS_MODEL is not set — point it at a tool-capable GGUF", file=sys.stderr)
        return 1

    modelnexus.set_log_level(modelnexus.LogLevel.ERROR)

    # A deliberately small window, so the budget is something you can watch fill up.
    with modelnexus.Chat(path, n_ctx=2048) as chat:
        messages = [
            {"role": "system", "content": "You are a support agent for a bicycle shop."},
            {"role": "user", "content": "My rear derailleur skips under load in the two lowest gears."},
        ]

        # Grow the conversation and watch the count against the window. This is the loop
        # a real agent runs before every call, to decide whether to trim history.
        print("messages   tokens   n_ctx   used")
        for _ in range(5):
            count = chat.count_tokens(messages)
            print(
                f"{len(messages):8d} {count['tokens']:8d} {count['n_ctx']:7d} "
                f"{100 * count['tokens'] / count['n_ctx']:5.1f}%"
            )
            messages += [
                {
                    "role": "assistant",
                    "content": "Check the cable tension at the barrel adjuster and index "
                    "the shifter again. " * 6,
                },
                {"role": "user", "content": "That did not fix it. What else?"},
            ]

        bare = chat.count_tokens(messages[:2])
        with_tools = chat.count_tokens(messages[:2], tools=TOOLS)

    print()
    print(
        f"the same two messages cost {bare['tokens']} tokens, or {with_tools['tokens']} "
        f"once one tool declaration is attached (+{with_tools['tokens'] - bare['tokens']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
