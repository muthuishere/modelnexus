"""conversation -- a multi-turn loop, with per-turn wall clock printed.

Every turn resends the whole conversation, so the prompt grows without bound. The
engine keeps what is already in its KV cache and re-decodes only the part that differs,
which for an appending conversation is just the new turn. The cost of re-reading the
prefix therefore stops growing.

This example does not assert a speedup. It runs the SAME turns twice — once with reuse
on (the default) and once with reuse_cache=False — and prints both clocks so the reader
sees whatever this machine actually does.

    MODELNEXUS_MODEL=/path/to/model.gguf python3 conversation.py
"""

import os
import sys
import time

import modelnexus

# Scripted turns, so both runs do identical work and the only variable is the cache.
TURNS = [
    "I am planning a week in Lisbon. Give me one neighbourhood to stay in.",
    "What is one dish I should eat there?",
    "Name one day trip within two hours.",
    "What is the weather like in October?",
    "One phrase of Portuguese I should learn?",
    "Is the metro worth using?",
    "One museum worth an afternoon?",
    "Sum up the trip in one sentence.",
]

# A system prompt long enough that the reused prefix is worth something from turn one.
SYSTEM = (
    "You are a concise travel assistant. Answer in at most two short sentences. "
    "Never list more than one option. Do not repeat the question back. Do not add "
    "caveats about checking current information. Assume the traveller is an experienced "
    "adult who has been to Europe before and wants opinions, not disclaimers."
)


def run(path: str, reuse: bool):
    """One full conversation. Returns per-turn milliseconds and prompt sizes."""
    elapsed, prompts = [], []
    with modelnexus.Chat(path) as chat:
        messages = [{"role": "system", "content": SYSTEM}]
        for question in TURNS:
            messages.append({"role": "user", "content": question})

            start = time.perf_counter()
            reply = chat.infer(
                messages=messages,
                # reuse_cache defaults to the core's own default (reuse on). Passing
                # None would be indistinguishable from not passing it, so this example
                # states it explicitly on both sides.
                reuse_cache=reuse,
                temperature=0.0,
                seed=11,
                max_tokens=40,
            )
            elapsed.append((time.perf_counter() - start) * 1000)
            prompts.append(reply["usage"]["prompt_tokens"])

            # Appending the reply is what makes the next prompt a strict extension of
            # this one — exactly the shape prefix reuse can exploit.
            messages.append({"role": "assistant", "content": reply["text"]})
    return elapsed, prompts


def main() -> int:
    path = os.environ.get("MODELNEXUS_MODEL")
    if not path:
        print("MODELNEXUS_MODEL is not set — point it at a tool-capable GGUF", file=sys.stderr)
        return 1

    modelnexus.set_log_level(modelnexus.LogLevel.ERROR)

    # Each run gets a fresh engine so neither inherits the other's cache.
    reuse_ms, prompts = run(path, reuse=True)
    fresh_ms, _ = run(path, reuse=False)

    print()
    print("turn  prompt tokens   reuse on   reuse off")
    for i, _ in enumerate(TURNS):
        print(f"{i + 1:3d} {prompts[i]:12d} {reuse_ms[i]:8.0f} ms {fresh_ms[i]:8.0f} ms")
    print(f"total {sum(reuse_ms):19.0f} ms {sum(fresh_ms):8.0f} ms")
    print()

    # Wall clock, not a claim: these numbers are whatever this machine produced just
    # now. Each turn generates the same 40 tokens, so the difference between the columns
    # is the prefill the reuse run did not have to redo.
    print(f"prompt grew from {prompts[0]} to {prompts[-1]} tokens across {len(TURNS)} turns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
