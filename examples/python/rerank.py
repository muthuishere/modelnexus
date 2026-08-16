"""rerank -- score documents against a query with a reranker model, best first.

A reranker reads the query and the document TOGETHER and emits one relevance logit,
which is why it beats comparing two independently-computed embeddings — and why it
costs a forward pass per document. The usual shape is: retrieve 50 by embedding, rerank
them, keep 5.

This needs a reranker GGUF, not a chat model, opened with pooling="rank": that is what
attaches the model's classification head to the graph. Without it the call fails with
POOLING_NOT_RANK rather than returning numbers that look like scores.

    MODELNEXUS_RERANKER=/path/to/bge-reranker.gguf python3 rerank.py
"""

import os
import sys

import modelnexus

QUERY = "How do I stop my sourdough loaf from being dense?"

DOCUMENTS = [
    "Sourdough stays dense when the starter is underactive: feed it twice daily until it doubles in four hours.",
    "The Rialto Bridge in Venice was completed in 1591 and spans the Grand Canal.",
    "Under-proofed dough has not built enough gas; extend the bulk ferment until the dough rises by half.",
    "Store flour in an airtight container away from sunlight to keep weevils out.",
    "Over-handling during shaping degasses the crumb, giving a tight, heavy loaf.",
]


def main() -> int:
    path = os.environ.get("MODELNEXUS_RERANKER")
    if not path:
        print("MODELNEXUS_RERANKER is not set — point it at a reranker GGUF", file=sys.stderr)
        return 1

    modelnexus.set_log_level(modelnexus.LogLevel.ERROR)

    with modelnexus.Embedder(path, pooling="rank") as rr:
        # Omitting top_n returns every document. Results arrive sorted best-first.
        hits = rr.rerank(QUERY, DOCUMENTS)

    print(f"query: {QUERY}\n")
    print("rank   score   original   document")
    for rank, hit in enumerate(hits, start=1):
        # "index" is the document's position in the ORIGINAL list. The list came back
        # reordered, so this is the only way to map a score onto your own data.
        print(f"{rank:4d} {hit['score']:8.3f} {hit['index']:10d}   {DOCUMENTS[hit['index']]}")

    print()
    print("Scores are raw model logits: comparable inside this one call, not across")
    print("models, and not probabilities.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
