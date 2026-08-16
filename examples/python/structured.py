"""structured -- pass a JSON Schema, get output that is guaranteed to parse.

A schema is compiled into a grammar that constrains decoding, so the model cannot emit
a token that would break the shape. The usual small-model failure — your JSON plus an
apology, or a truncated object — becomes impossible rather than unlikely.

    MODELNEXUS_MODEL=/path/to/model.gguf python3 structured.py
"""

import json
import os
import sys

import modelnexus

# "required" and "enum" are worth setting: they are constraints the grammar can
# enforce, which makes them free — unlike a prompt instruction the model may ignore.
SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "mixed"]},
        "rating": {"type": "integer", "minimum": 1, "maximum": 5},
        "topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sentiment", "rating", "topics"],
    "additionalProperties": False,
}

REVIEW = (
    'Classify this review: "The battery lasts two days and the screen is gorgeous, '
    'but the camera is mediocre and it costs too much."'
)


def main() -> int:
    path = os.environ.get("MODELNEXUS_MODEL")
    if not path:
        print("MODELNEXUS_MODEL is not set — point it at a tool-capable GGUF", file=sys.stderr)
        return 1

    modelnexus.set_log_level(modelnexus.LogLevel.ERROR)

    with modelnexus.Chat(path) as chat:
        reply = chat.infer(
            messages=[{"role": "user", "content": REVIEW}],
            json_schema=SCHEMA,
            temperature=0.0,
            seed=7,
            max_tokens=120,
        )

    # No repair pass, no fence stripping, no retry loop. The core already removed the
    # ```json fence llama.cpp's generated grammar permits, so what arrives is JSON.
    try:
        parsed = json.loads(reply["text"])
    except json.JSONDecodeError as exc:
        print("the schema did not hold:", exc, file=sys.stderr)
        print("raw:", reply["text"], file=sys.stderr)
        return 1

    print("sentiment:", parsed["sentiment"])
    print("rating:   ", parsed["rating"])
    print("topics:   ", parsed["topics"])
    print(
        f"(parsed from {len(reply['text'])} bytes of model output, "
        f"finish_reason={reply['finish_reason']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
