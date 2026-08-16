"""embeddings -- turn sentences into vectors, then compare them.

Vectors come back L2-normalised, so a dot product IS the cosine similarity. There is no
norm to divide by and no numpy to import for it: the one-liner below is the whole of
the maths.

An Embedder is a separate handle from a Chat on purpose — embedding needs a context
built with embeddings enabled and a pooling strategy fixed at creation, neither of
which can be switched on a generation context afterwards.

    MODELNEXUS_MODEL=/path/to/model.gguf python3 embeddings.py
"""

import os
import sys

import modelnexus

SENTENCES = [
    "The cat sat on the mat.",
    "A kitten rested on the rug.",
    "Interest rates rose again this quarter.",
    "The central bank raised borrowing costs.",
]


def cosine(a, b) -> float:
    """A plain dot product, valid only because embed() normalises. Use
    normalize=False and you owe yourself the division."""
    return sum(x * y for x, y in zip(a, b))


def main() -> int:
    path = os.environ.get("MODELNEXUS_MODEL")
    if not path:
        print("MODELNEXUS_MODEL is not set — point it at a GGUF", file=sys.stderr)
        return 1

    modelnexus.set_log_level(modelnexus.LogLevel.ERROR)

    # Mean pooling averages the token vectors — the usual choice for sentence
    # similarity. A dedicated embedding model would be smaller and better; a chat model
    # works and keeps this example to one model file.
    with modelnexus.Embedder(path, pooling="mean") as emb:
        # One call, several inputs: they are batched into as few decodes as n_batch
        # allows, and come back in input order.
        vectors = emb.embed(SENTENCES)

    print(f"{len(vectors)} vectors of {len(vectors[0])} dimensions\n")
    for i, s in enumerate(SENTENCES):
        print(f"  [{i}] {s}")
    print()

    print("cosine   " + "".join(f"{i:8d}" for i in range(len(SENTENCES))))
    for i in range(len(SENTENCES)):
        row = "".join(f"{cosine(vectors[i], vectors[j]):8.3f}" for j in range(len(SENTENCES)))
        print(f"{i:6d}   {row}")

    print()
    print(f"cats  [0]x[1] = {cosine(vectors[0], vectors[1]):.3f}")
    print(f"rates [2]x[3] = {cosine(vectors[2], vectors[3]):.3f}")
    print(f"across topics [0]x[2] = {cosine(vectors[0], vectors[2]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
