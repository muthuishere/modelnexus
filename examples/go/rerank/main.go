// rerank -- score documents against a query with a reranker model, best first.
//
// A reranker reads the query and the document TOGETHER and emits one relevance logit,
// which is why it beats comparing two independently-computed embeddings — and why it
// costs a forward pass per document. The usual shape is: retrieve 50 by embedding,
// rerank them, keep 5.
//
// This needs a reranker GGUF, not a chat model, opened with "rank" pooling: that is
// what attaches the model's classification head to the graph. Without it the call
// fails with POOLING_NOT_RANK rather than returning numbers that look like scores.
//
//	MODELNEXUS_RERANKER=/path/to/bge-reranker.gguf go run ./rerank
package main

import (
	"fmt"
	"os"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

const query = "How do I stop my sourdough loaf from being dense?"

var documents = []string{
	"Sourdough stays dense when the starter is underactive: feed it twice daily until it doubles in four hours.",
	"The Rialto Bridge in Venice was completed in 1591 and spans the Grand Canal.",
	"Under-proofed dough has not built enough gas; extend the bulk ferment until the dough rises by half.",
	"Store flour in an airtight container away from sunlight to keep weevils out.",
	"Over-handling during shaping degasses the crumb, giving a tight, heavy loaf.",
}

func main() {
	path := os.Getenv("MODELNEXUS_RERANKER")
	if path == "" {
		fmt.Fprintln(os.Stderr, "MODELNEXUS_RERANKER is not set — point it at a reranker GGUF")
		os.Exit(1)
	}

	if err := modelnexus.SetLogLevel(modelnexus.LogError); err != nil {
		fmt.Fprintln(os.Stderr, "log level:", err)
		os.Exit(1)
	}

	rr, err := modelnexus.OpenEmbedder(path, &modelnexus.EmbedOptions{
		Pooling: modelnexus.PoolingRank,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "open reranker:", err)
		os.Exit(1)
	}
	defer rr.Close()

	// topN <= 0 returns every document. Results arrive sorted best-first.
	hits, err := rr.Rerank(query, documents, 0)
	if err != nil {
		fmt.Fprintln(os.Stderr, "rerank:", err)
		os.Exit(1)
	}

	fmt.Printf("query: %s\n\n", query)
	fmt.Println("rank   score   original   document")
	for rank, hit := range hits {
		// Index is the document's position in the ORIGINAL slice. The list came back
		// reordered, so this is the only way to map a score onto your own data.
		fmt.Printf("%4d %8.3f %10d   %s\n", rank+1, hit.Score, hit.Index, documents[hit.Index])
	}

	fmt.Println()
	fmt.Println("Scores are raw model logits: comparable inside this one call, not across")
	fmt.Println("models, and not probabilities.")
}
