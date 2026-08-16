// embeddings -- turn sentences into vectors, then compare them.
//
// Vectors come back L2-normalised, so a dot product IS the cosine similarity. There is
// no norm to divide by and no library to pull in for it: the loop below is the whole
// of the maths.
//
// An Embedder is a separate handle from a Chat on purpose — embedding needs a context
// built with embeddings enabled and a pooling strategy fixed at creation, neither of
// which can be switched on a generation context afterwards.
//
//	MODELNEXUS_MODEL=/path/to/model.gguf go run ./embeddings
package main

import (
	"fmt"
	"os"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

var sentences = []string{
	"The cat sat on the mat.",
	"A kitten rested on the rug.",
	"Interest rates rose again this quarter.",
	"The central bank raised borrowing costs.",
}

// cosine is a plain dot product, which is only valid because Embed normalises. Use
// EmbedRaw and you owe yourself the division.
func cosine(a, b []float32) float64 {
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

func main() {
	path := os.Getenv("MODELNEXUS_MODEL")
	if path == "" {
		fmt.Fprintln(os.Stderr, "MODELNEXUS_MODEL is not set — point it at a GGUF")
		os.Exit(1)
	}

	if err := modelnexus.SetLogLevel(modelnexus.LogError); err != nil {
		fmt.Fprintln(os.Stderr, "log level:", err)
		os.Exit(1)
	}

	// Mean pooling averages the token vectors — the usual choice for sentence
	// similarity. A dedicated embedding model would be smaller and better; a chat
	// model works and keeps this example to one model file.
	emb, err := modelnexus.OpenEmbedder(path, &modelnexus.EmbedOptions{
		Pooling: modelnexus.PoolingMean,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "open embedder:", err)
		os.Exit(1)
	}
	defer emb.Close()

	// One call, several inputs: they are batched into as few decodes as n_batch allows,
	// and come back in input order.
	vectors, err := emb.Embed(sentences)
	if err != nil {
		fmt.Fprintln(os.Stderr, "embed:", err)
		os.Exit(1)
	}

	fmt.Printf("%d vectors of %d dimensions\n\n", len(vectors), len(vectors[0]))
	for i, s := range sentences {
		fmt.Printf("  [%d] %s\n", i, s)
	}
	fmt.Println()

	fmt.Print("cosine   ")
	for i := range sentences {
		fmt.Printf("%8d", i)
	}
	fmt.Println()
	for i := range sentences {
		fmt.Printf("%6d   ", i)
		for j := range sentences {
			fmt.Printf("%8.3f", cosine(vectors[i], vectors[j]))
		}
		fmt.Println()
	}

	fmt.Println()
	fmt.Printf("cats  [0]x[1] = %.3f\n", cosine(vectors[0], vectors[1]))
	fmt.Printf("rates [2]x[3] = %.3f\n", cosine(vectors[2], vectors[3]))
	fmt.Printf("across topics [0]x[2] = %.3f\n", cosine(vectors[0], vectors[2]))
}
