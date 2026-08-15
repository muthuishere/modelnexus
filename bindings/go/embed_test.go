package modelnexus_test

import (
	"math"
	"os"
	"testing"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

func reranker(t *testing.T) string {
	t.Helper()
	path := os.Getenv("MODELNEXUS_RERANKER")
	if path == "" {
		t.Skip("set MODELNEXUS_RERANKER to a reranker GGUF to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("MODELNEXUS_RERANKER=%s is not readable: %v", path, err)
	}
	return path
}

func TestLoRALifecycle(t *testing.T) {
	chat, err := modelnexus.Open(model(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer chat.Close()

	got, err := chat.LoRAs()
	if err != nil {
		t.Fatalf("LoRAs: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected no adapters on a fresh engine, got %d", len(got))
	}

	// Clearing an empty set must be a no-op, not an error -- callers should not
	// have to check first.
	if err := chat.ClearLoRAs(); err != nil {
		t.Errorf("ClearLoRAs on an empty set: %v", err)
	}
}

func TestLoRAErrorsAreTyped(t *testing.T) {
	chat, err := modelnexus.Open(model(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer chat.Close()

	if _, err := chat.LoadLoRA("/definitely/not/here.gguf", 1.0); err == nil {
		t.Error("expected an error loading a missing adapter")
	} else {
		var me *modelnexus.Error
		if !errorsAs(err, &me) || me.Code != "LORA_LOAD_FAILED" {
			t.Errorf("got %v, want code LORA_LOAD_FAILED", err)
		}
	}

	if err := chat.SetLoRAScale(99, 0.5); err == nil {
		t.Error("expected an error setting a scale on an unknown id")
	} else {
		var me *modelnexus.Error
		if !errorsAs(err, &me) || me.Code != "LORA_NOT_FOUND" {
			t.Errorf("got %v, want code LORA_NOT_FOUND", err)
		}
	}
}

func TestEmbed(t *testing.T) {
	emb, err := modelnexus.OpenEmbedder(model(t), &modelnexus.EmbedOptions{
		Pooling: modelnexus.PoolingMean,
		NCtx:    512,
	})
	if err != nil {
		t.Fatalf("OpenEmbedder: %v", err)
	}
	defer emb.Close()

	vectors, err := emb.Embed([]string{"hello world", "goodbye world"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vectors) != 2 {
		t.Fatalf("got %d vectors, want 2", len(vectors))
	}
	if len(vectors[0]) == 0 || len(vectors[0]) != len(vectors[1]) {
		t.Fatalf("inconsistent dimensions: %d and %d", len(vectors[0]), len(vectors[1]))
	}

	// Embed normalizes, so every vector is unit length and a dot product is a
	// cosine similarity. If this drifts, every downstream similarity silently
	// changes meaning.
	var sum float64
	for _, v := range vectors[0] {
		sum += float64(v) * float64(v)
	}
	if norm := math.Sqrt(sum); math.Abs(norm-1.0) > 1e-3 {
		t.Errorf("L2 norm = %f, want 1.0", norm)
	}
}

func TestEmbedRawIsNotNormalized(t *testing.T) {
	emb, err := modelnexus.OpenEmbedder(model(t), &modelnexus.EmbedOptions{
		Pooling: modelnexus.PoolingMean,
		NCtx:    512,
	})
	if err != nil {
		t.Fatalf("OpenEmbedder: %v", err)
	}
	defer emb.Close()

	raw, err := emb.EmbedRaw([]string{"hello world"})
	if err != nil {
		t.Fatalf("EmbedRaw: %v", err)
	}
	var sum float64
	for _, v := range raw[0] {
		sum += float64(v) * float64(v)
	}
	if norm := math.Sqrt(sum); math.Abs(norm-1.0) < 1e-6 {
		t.Error("EmbedRaw returned a unit vector; normalization was not skipped")
	}
}

func TestRerankRefusesWithoutRankPooling(t *testing.T) {
	// Returning plausible-looking numbers from a model whose classification head is
	// not even in the graph would be far worse than failing.
	emb, err := modelnexus.OpenEmbedder(model(t), &modelnexus.EmbedOptions{
		Pooling: modelnexus.PoolingMean,
		NCtx:    512,
	})
	if err != nil {
		t.Fatalf("OpenEmbedder: %v", err)
	}
	defer emb.Close()

	if _, err := emb.Rerank("q", []string{"a"}, 0); err == nil {
		t.Fatal("expected rerank to refuse a non-rank engine")
	} else {
		var me *modelnexus.Error
		if !errorsAs(err, &me) || me.Code != "POOLING_NOT_RANK" {
			t.Errorf("got %v, want code POOLING_NOT_RANK", err)
		}
	}
}

func TestRerankRanksSemantically(t *testing.T) {
	emb, err := modelnexus.OpenEmbedder(reranker(t), &modelnexus.EmbedOptions{
		Pooling: modelnexus.PoolingRank,
		NCtx:    512,
	})
	if err != nil {
		t.Fatalf("OpenEmbedder: %v", err)
	}
	defer emb.Close()

	docs := []string{
		"Berlin is the capital and largest city of Germany.",
		"Paris has been France's capital since the 10th century.",
		"Bananas are a good source of potassium.",
	}
	hits, err := emb.Rerank("What is the capital of France?", docs, 0)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(hits) != len(docs) {
		t.Fatalf("got %d hits, want %d", len(hits), len(docs))
	}
	// The Paris document must win, and results must be sorted best-first.
	if hits[0].Index != 1 {
		t.Errorf("top hit index = %d, want 1 (the Paris document)", hits[0].Index)
	}
	for i := 1; i < len(hits); i++ {
		if hits[i].Score > hits[i-1].Score {
			t.Errorf("results are not sorted descending at %d", i)
		}
	}

	top, err := emb.Rerank("What is the capital of France?", docs, 1)
	if err != nil {
		t.Fatalf("Rerank top_n: %v", err)
	}
	if len(top) != 1 {
		t.Errorf("top_n=1 returned %d hits", len(top))
	}
}

// loraPair returns a base model and a LoRA adapter built for it, or skips.
//
// A LoRA is architecture-specific -- it only loads against the base it was trained
// on -- so this needs its own matched pair rather than reusing MODELNEXUS_MODEL.
func loraPair(t *testing.T) (string, string) {
	t.Helper()
	base := os.Getenv("MODELNEXUS_LORA_BASE")
	lora := os.Getenv("MODELNEXUS_LORA")
	if base == "" || lora == "" {
		t.Skip("set MODELNEXUS_LORA_BASE and MODELNEXUS_LORA to run this test")
	}
	for _, p := range []string{base, lora} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("%s is not readable: %v", p, err)
		}
	}
	return base, lora
}

func TestLoRAFullLifecycleWithARealAdapter(t *testing.T) {
	base, lora := loraPair(t)
	chat, err := modelnexus.Open(base)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer chat.Close()

	first, err := chat.LoadLoRA(lora, 1.0)
	if err != nil {
		t.Fatalf("LoadLoRA: %v", err)
	}
	applied, err := chat.LoRAs()
	if err != nil {
		t.Fatalf("LoRAs: %v", err)
	}
	if len(applied) != 1 || applied[0].ID != first || applied[0].Scale != 1.0 {
		t.Fatalf("unexpected adapter state: %+v", applied)
	}

	if err := chat.SetLoRAScale(first, 0.5); err != nil {
		t.Fatalf("SetLoRAScale: %v", err)
	}
	if applied, _ = chat.LoRAs(); applied[0].Scale != 0.5 {
		t.Errorf("scale = %v, want 0.5", applied[0].Scale)
	}

	// Several adapters can be active at once; ids stay stable and independent.
	second, err := chat.LoadLoRA(lora, 0.25)
	if err != nil {
		t.Fatalf("second LoadLoRA: %v", err)
	}
	if applied, _ = chat.LoRAs(); len(applied) != 2 {
		t.Fatalf("expected 2 adapters, got %d", len(applied))
	}

	if err := chat.RemoveLoRA(second); err != nil {
		t.Fatalf("RemoveLoRA: %v", err)
	}
	if applied, _ = chat.LoRAs(); len(applied) != 1 || applied[0].ID != first {
		t.Fatalf("after remove: %+v", applied)
	}

	// Generation must still work with an adapter applied -- the point of loading it.
	max := 12
	seed := uint32(1)
	resp, err := chat.Infer(modelnexus.Request{
		Messages:  []modelnexus.Message{{Role: "user", Content: "Say the word: apple"}},
		MaxTokens: &max, Seed: &seed,
	})
	if err != nil {
		t.Fatalf("Infer with adapter: %v", err)
	}
	if resp.Text == "" {
		t.Error("expected text with an adapter applied")
	}

	if err := chat.ClearLoRAs(); err != nil {
		t.Fatalf("ClearLoRAs: %v", err)
	}
	if applied, _ = chat.LoRAs(); len(applied) != 0 {
		t.Errorf("expected no adapters after clear, got %d", len(applied))
	}
}
