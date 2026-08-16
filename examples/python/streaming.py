"""streaming -- print tokens as they arrive, and stop early from the callback.

Stopping is the point. Before 0.2.0 a consumer who walked away — a closed stream, a
user pressing stop — could not tell the model, so it generated to completion and you
paid for all of it. Returning False from on_token ends the turn now.

    MODELNEXUS_MODEL=/path/to/model.gguf python3 streaming.py
"""

import os
import sys

import modelnexus

BUDGET = 20


def main() -> int:
    path = os.environ.get("MODELNEXUS_MODEL")
    if not path:
        print("MODELNEXUS_MODEL is not set — point it at a tool-capable GGUF", file=sys.stderr)
        return 1

    # Quieten the engine so the streamed tokens are the only thing on the terminal.
    modelnexus.set_log_level(modelnexus.LogLevel.ERROR)

    max_tokens = 200  # deliberately more than we intend to read
    seen = 0

    def on_token(piece: str):
        nonlocal seen
        print(piece, end="", flush=True)
        seen += 1
        # Only an explicit False stops. Returning None — which is what a callback that
        # merely prints does — keeps going, so a print-only stream is not accidentally
        # cancelled after its first token.
        if seen >= BUDGET:
            return False
        return None

    print("streaming: ", end="", flush=True)
    with modelnexus.Chat(path) as chat:
        reply = chat.infer(
            messages=[{"role": "user", "content": "List the planets of the solar system, one per line."}],
            on_token=on_token,
            temperature=0.0,
            seed=42,
            max_tokens=max_tokens,
        )
    print()

    # A cancelled generation is a RESULT, not an error: the response is complete, the
    # text is what was really produced, and the usage counts are the tokens you were
    # really charged for. Nothing raised above, precisely because nothing went wrong.
    usage = reply["usage"]
    print()
    print("finish_reason:", reply["finish_reason"])
    print("cancelled:    ", reply["finish_reason"] == "cancelled")
    print(
        f"usage:         {usage['prompt_tokens']} prompt + {usage['completion_tokens']} "
        f"completion (asked for up to {max_tokens})"
    )
    print(f"pieces seen:   {seen}, response text length: {len(reply['text'])} bytes")

    if reply["finish_reason"] != "cancelled":
        print("expected the callback to have stopped generation", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
