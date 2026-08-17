// citenexus with modelnexus behind it: embeddings AND generation, no network.
//
// citenexus's seam is one function type (bring-your-own-model):
//
//	type Transport func(url string, body []byte, headers map[string]string) ([]byte, error)
//
// It is the OpenAI wire shape, which is exactly where modelnexus already meets
// everything else (ADR-0003). So ONE adapter serves both the embedder and the
// generator -- a nicer fit than a per-port HTTP client, because there is no HTTP
// vocabulary to fake.
//
//	MODELNEXUS_MODEL=chat.gguf MODELNEXUS_EMBED=embed.gguf go run ./citenexus
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
	"github.com/muthuishere/citenexus/golang/models"
)

// localTransport answers citenexus's model calls in this process. It branches on
// the URL because that is the ONLY thing distinguishing an embedding request
// from a completion in the OpenAI shape -- the bodies differ, but the endpoint
// is what citenexus tells you.
type localTransport struct {
	chat  *modelnexus.Chat
	embed *modelnexus.Embedder
}

func (t *localTransport) Do(url string, body []byte, _ map[string]string) ([]byte, error) {
	if strings.Contains(url, "/embeddings") {
		return t.embeddings(body)
	}
	return t.completions(body)
}

func (t *localTransport) embeddings(body []byte) ([]byte, error) {
	// "input" is a string OR an array of strings. Handling only the array is the
	// classic way this adapter half-works.
	var req struct {
		Input json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	var texts []string
	if err := json.Unmarshal(req.Input, &texts); err != nil {
		var one string
		if err2 := json.Unmarshal(req.Input, &one); err2 != nil {
			return nil, fmt.Errorf("input is neither a string nor an array: %w", err)
		}
		texts = []string{one}
	}

	vecs, err := t.embed.Embed(texts)
	if err != nil {
		return nil, err
	}
	data := make([]any, 0, len(vecs))
	for i, v := range vecs {
		data = append(data, map[string]any{"object": "embedding", "index": i, "embedding": v})
	}
	return json.Marshal(map[string]any{"object": "list", "data": data})
}

func (t *localTransport) completions(body []byte) ([]byte, error) {
	var req struct {
		Messages []modelnexus.Message `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	temp, seed, maxTok := 0.0, uint32(7), 256
	res, err := t.chat.Infer(modelnexus.Request{
		Messages: req.Messages, Temperature: &temp, Seed: &seed, MaxTokens: &maxTok,
	})
	if err != nil {
		return nil, err
	}
	return json.Marshal(map[string]any{
		"choices": []any{map[string]any{
			"message":       map[string]any{"role": "assistant", "content": res.Text},
			"finish_reason": res.FinishReason,
		}},
		"usage": map[string]any{
			"prompt_tokens": res.Usage.PromptTokens, "completion_tokens": res.Usage.CompletionTokens,
			"total_tokens": res.Usage.TotalTokens,
		},
	})
}

func main() {
	chatPath, embedPath := os.Getenv("MODELNEXUS_MODEL"), os.Getenv("MODELNEXUS_EMBED")
	if chatPath == "" || embedPath == "" {
		fmt.Fprintln(os.Stderr, "set MODELNEXUS_MODEL (chat) and MODELNEXUS_EMBED (embedding) GGUFs")
		os.Exit(1)
	}
	_ = modelnexus.SetLogLevel(modelnexus.LogError)

	chat, err := modelnexus.Open(chatPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open chat:", err)
		os.Exit(1)
	}
	defer chat.Close()

	emb, err := modelnexus.OpenEmbedder(embedPath, &modelnexus.EmbedOptions{Pooling: modelnexus.PoolingMean})
	if err != nil {
		fmt.Fprintln(os.Stderr, "open embedder:", err)
		os.Exit(1)
	}
	defer emb.Close()

	lt := &localTransport{chat: chat, embed: emb}

	// baseURL is never dialled -- it only tells the adapter which call this is.
	embedder := models.NewOpenAIEmbedding("http://local/v1", "modelnexus", lt.Do)
	generator := models.NewOpenAIChatGenerator("http://local/v1", "modelnexus", 0.0, nil, lt.Do)

	fmt.Println("embedding through citenexus, model in this process...")
	vecs, err := embedder.Embed([]string{
		"Sourdough stays dense when the starter is underactive.",
		"The Rialto Bridge in Venice was completed in 1591.",
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "embed:", err)
		os.Exit(1)
	}
	fmt.Printf("  %d vectors of %d dimensions\n", len(vecs), len(vecs[0]))

	// citenexus's generator is GROUNDED by construction: a question plus the
	// passage it must answer from. That is the whole product -- an answer you can
	// cite -- and it is why the seam is worth wiring to a local model rather than
	// a hosted one, since the passages are usually the sensitive part.
	fmt.Println("\nanswering from a passage, through citenexus...")
	passage := "Sourdough stays dense when the starter is underactive: feed it twice " +
		"daily until it doubles in four hours."
	out, err := generator.Answer("Why is my sourdough dense?", passage, "en")
	if err != nil {
		fmt.Fprintln(os.Stderr, "answer:", err)
		os.Exit(1)
	}
	fmt.Println("  answer:", strings.TrimSpace(out))
	fmt.Println("\nboth seams served by one adapter; no socket was opened.")
}
