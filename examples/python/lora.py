"""lora -- apply a LoRA adapter to a live engine, then take it off again.

Adapters load against the model already in memory: no reload, no second copy of the
weights, and several can be active at once with independent scales. They change
*behaviour* — tone, output format, tool-call reliability — not knowledge. For facts,
retrieve.

The adapter and the base model are a matched PAIR: an adapter is built for one
architecture and one tensor layout, and will not load against an arbitrary GGUF. Hence
two env vars rather than reusing MODELNEXUS_MODEL.

    MODELNEXUS_LORA_BASE=/path/to/base.gguf MODELNEXUS_LORA=/path/to/adapter.gguf python3 lora.py
"""

import os
import sys

import modelnexus

# The adapter used to develop this example removes the base model's refusal behaviour,
# so a prompt the base declines is the one place the difference is legible.
PROMPT = "Say something rude about the weather in one sentence."

# Scale is a dial, not a switch. This adapter is f16 against a q4 base, and at 1.0 it
# overwhelms the model — output degenerates into fragments. 0.25 shifts behaviour and
# keeps the model coherent. Any adapter you did not train yourself deserves this sweep.
SCALE = 0.25


def main() -> int:
    base = os.environ.get("MODELNEXUS_LORA_BASE")
    adapter = os.environ.get("MODELNEXUS_LORA")
    if not base or not adapter:
        print(
            "set MODELNEXUS_LORA_BASE and MODELNEXUS_LORA to a matched base/adapter pair",
            file=sys.stderr,
        )
        return 1

    modelnexus.set_log_level(modelnexus.LogLevel.ERROR)

    with modelnexus.Chat(base) as chat:
        # Temperature 0 and a fixed seed, so the only thing that can move the output
        # between the three calls below is the adapter.
        def ask() -> str:
            return chat.infer(
                messages=[{"role": "user", "content": PROMPT}],
                temperature=0.0,
                seed=3,
                max_tokens=60,
                # Each call must be provably independent, or the previous call's KV
                # prefix — computed under a different adapter set — could be reused
                # underneath it.
                reuse_cache=False,
            )["text"]

        print("prompt:", PROMPT)
        print()

        before = ask()
        print("--- base model ---")
        print(before)
        print()

        adapter_id = chat.load_lora(adapter, scale=SCALE)
        applied = chat.loras()
        print(f"--- adapter {adapter_id} applied at scale {applied[0]['scale']:.2f} ---")
        during = ask()
        print(during)
        print()

        # clear_loras unloads every adapter and reapplies nothing, so the engine is back
        # to the weights it loaded from disk.
        chat.clear_loras()
        after = ask()
        print("--- adapter cleared ---")
        print(after)
        print()

    if before != after:
        # This is the check worth failing on: removing an adapter must restore the base
        # model exactly, or the engine is carrying state it should not.
        print("clearing the adapter did not restore the base model's output", file=sys.stderr)
        return 1

    print("clearing restored the base output byte for byte.")
    if before == during:
        print("this adapter did not change the answer to this particular prompt.")
    else:
        print("the adapter changed the answer; the base model was restored by clearing it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
