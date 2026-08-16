"""hello -- load a GGUF, run one inference, print the answer.

The smallest thing that works. There is no server to start, no port to pick and no
subprocess: the model is mapped into this interpreter and decodes in it.

    MODELNEXUS_MODEL=/path/to/model.gguf python3 hello.py
"""

import os
import sys

import modelnexus


def main() -> int:
    path = os.environ.get("MODELNEXUS_MODEL")
    if not path:
        print("MODELNEXUS_MODEL is not set — point it at a tool-capable GGUF", file=sys.stderr)
        return 1

    # llama.cpp narrates the load. The bridge already owns that sink and defaults to
    # WARN; dropping to ERROR leaves this program's own output as the only output.
    # Set it before loading — logging starts during the load, so afterwards is too late.
    modelnexus.set_log_level(modelnexus.LogLevel.ERROR)

    # Chat loads the weights and builds the inference context. It rejects a model whose
    # chat template cannot do tool calling, rather than loading it and degrading tool
    # calls to prose later — the failure arrives here, where it is cheap to read.
    #
    # The handle owns native memory, so it is a context manager rather than something
    # the garbage collector gets to close whenever it likes.
    with modelnexus.Chat(path) as chat:
        # Generation parameters travel inside the request, so a new one in the core
        # needs no change in this binding: they are keyword arguments passed through.
        reply = chat.infer(
            messages=[{"role": "user", "content": "Name the capital of France. Answer in one word."}],
            temperature=0.0,
            seed=42,
            max_tokens=64,
        )

    print("answer:", reply["text"])
    usage = reply["usage"]
    print(
        f"tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion, "
        f"finish_reason={reply['finish_reason']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
