package modelnexus_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

// The Go mirror of core/tests/abi_test.c. Three of these exist because the spike
// behind ADR-0008 found the failure mode before the feature shipped, and each one is
// the kind of bug that is silent rather than loud.

// deterministic is the request shape every comparison test uses: one short question,
// temperature 0 and a fixed seed, so the only variable left is the thing under test.
func deterministic(prompt string, maxTokens int) modelnexus.Request {
	temp := 0.0
	seed := uint32(42)
	return modelnexus.Request{
		Messages:    []modelnexus.Message{{Role: "user", Content: prompt}},
		Temperature: &temp,
		Seed:        &seed,
		MaxTokens:   &maxTokens,
	}
}

func openChat(t *testing.T, opts ...modelnexus.Option) *modelnexus.Chat {
	t.Helper()
	chat, err := modelnexus.Open(model(t), opts...)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { chat.Close() })
	return chat
}

func TestOpenWithoutOptionsStillWorks(t *testing.T) {
	// The create signature gained a config parameter in 0.2.0. An Open with no
	// options must send an empty config and get exactly the old defaults.
	chat := openChat(t)
	max := 8
	if _, err := chat.Infer(modelnexus.Request{
		Messages:  []modelnexus.Message{{Role: "user", Content: "hi"}},
		MaxTokens: &max,
	}); err != nil {
		t.Fatalf("Infer on a default-configured engine: %v", err)
	}
}

func TestOpenWithExplicitContextConfig(t *testing.T) {
	chat := openChat(t,
		modelnexus.WithContextSize(2048),
		modelnexus.WithBatchSize(256),
		modelnexus.WithMaxSequences(1),
	)
	count, err := chat.CountTokens(deterministic("hello", 8))
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	// The context size we asked for is the one the engine reports -- the proof the
	// config crossed the ABI rather than being marshalled and dropped.
	if count.NCtx != 2048 {
		t.Errorf("n_ctx = %d, want the 2048 requested at open", count.NCtx)
	}
}

func TestCountTokens(t *testing.T) {
	chat := openChat(t)
	req := deterministic("Name the capital of France in one word.", 16)

	count, err := chat.CountTokens(req)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if count.Tokens <= 0 {
		t.Errorf("tokens = %d, want a positive prompt length", count.Tokens)
	}
	if count.NCtx <= 0 {
		t.Errorf("n_ctx = %d, want the engine's context window", count.NCtx)
	}
	if count.Tokens >= count.NCtx {
		t.Errorf("a nine-word prompt counted %d tokens against a %d window", count.Tokens, count.NCtx)
	}

	// A longer message list must count higher; a count that ignores its input would
	// still satisfy every bound above.
	longer := req
	longer.Messages = append(append([]modelnexus.Message(nil), req.Messages...),
		modelnexus.Message{Role: "assistant", Content: strings.Repeat("Paris. ", 40)})
	longerCount, err := chat.CountTokens(longer)
	if err != nil {
		t.Fatalf("CountTokens (longer): %v", err)
	}
	if longerCount.Tokens <= count.Tokens {
		t.Errorf("longer conversation counted %d tokens, not more than %d", longerCount.Tokens, count.Tokens)
	}

	// Counting must not disturb the cache it deliberately does not touch.
	if _, err := chat.Infer(req); err != nil {
		t.Fatalf("Infer after CountTokens: %v", err)
	}
}

func TestCacheReuseDoesNotChangeOutput(t *testing.T) {
	// Reuse is a latency property. Any observable difference in output is a defect,
	// and this is the assertion that catches it.
	chat := openChat(t)

	cold := deterministic("Name the capital of France in one word.", 16)
	cold.ReuseCache = modelnexus.ReuseCacheOff()
	warm := deterministic("Name the capital of France in one word.", 16)

	a, err := chat.Infer(cold) // cold, cache cleared
	if err != nil {
		t.Fatalf("Infer (no reuse): %v", err)
	}
	b, err := chat.Infer(warm) // warm, prefix reused
	if err != nil {
		t.Fatalf("Infer (reuse): %v", err)
	}
	c, err := chat.Infer(cold) // cold again -- the control
	if err != nil {
		t.Fatalf("Infer (no reuse, second): %v", err)
	}

	if a.Text != c.Text {
		t.Fatalf("repeated cold runs disagree, so the comparison below is meaningless:\n  %q\n  %q", a.Text, c.Text)
	}
	if a.Text != b.Text {
		t.Errorf("reuse changed the output:\n  reuse_cache=false: %q\n  reuse_cache=true:  %q", a.Text, b.Text)
	}
	if a.Usage.CompletionTokens != b.Usage.CompletionTokens {
		t.Errorf("completion tokens differ with reuse: %d vs %d", a.Usage.CompletionTokens, b.Usage.CompletionTokens)
	}
}

func TestCancelThenADifferentRequestIsStillCorrect(t *testing.T) {
	// The D2xD4 interaction: an abort leaves a partial assistant turn in the KV
	// cache, and without rollback the next call's prefix match extends a truncated
	// turn as though it were complete. Silent, plausible, wrong.
	chat := openChat(t)

	seen := 0
	long := deterministic("Count slowly from one to fifty in words, one per line.", 300)
	resp, err := chat.InferStream(long, func(string) bool {
		seen++
		return seen < 8 // stop once the eighth piece has been delivered
	})
	if err != nil {
		t.Fatalf("InferStream: %v", err)
	}
	if !resp.Cancelled() {
		t.Errorf("finish_reason = %q, want cancelled", resp.FinishReason)
	}
	if seen != 8 {
		t.Errorf("saw %d tokens, want generation to stop at 8", seen)
	}
	if resp.Usage.CompletionTokens != 8 {
		t.Errorf("usage reports %d completion tokens, want the 8 actually generated", resp.Usage.CompletionTokens)
	}

	after, err := chat.Infer(deterministic("Name the capital of France in one word.", 16))
	if err != nil {
		t.Fatalf("Infer after cancellation: %v", err)
	}
	if !strings.Contains(after.Text, "Paris") {
		t.Errorf("a request after a cancellation was wrong: %q", after.Text)
	}
}

func TestCallbackReturningTrueLetsGenerationFinish(t *testing.T) {
	chat := openChat(t)
	seen := 0
	resp, err := chat.InferStream(deterministic("Name the capital of France in one word.", 16), func(string) bool {
		seen++
		return true
	})
	if err != nil {
		t.Fatalf("InferStream: %v", err)
	}
	if resp.Cancelled() {
		t.Error("returning true cancelled generation")
	}
	if seen == 0 {
		t.Error("the callback saw no tokens")
	}
}

func TestCancelledContextStopsGeneration(t *testing.T) {
	// The Go affordance for the same ABI mechanism: no extra behaviour, just the
	// native idiom wired to the callback's return value.
	chat := openChat(t)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	seen := 0
	var streamed strings.Builder
	resp, err := chat.InferStreamContext(ctx, deterministic("Count slowly from one to fifty in words, one per line.", 300),
		func(piece string) bool {
			seen++
			streamed.WriteString(piece)
			if seen == 5 {
				cancel()
			}
			return true // the context, not the callback, is what stops this
		})
	if err != nil {
		t.Fatalf("InferStreamContext: %v", err)
	}
	if !resp.Cancelled() {
		t.Errorf("finish_reason = %q, want cancelled", resp.FinishReason)
	}
	// The context is checked after each piece is delivered, so cancelling from
	// inside the callback ends the run on that very token: five delivered, none
	// wasted. Cancelling from another goroutine instead lands before the next one.
	if seen != 5 {
		t.Errorf("saw %d tokens, want the run to end on the token that cancelled it", seen)
	}
	if resp.Usage.CompletionTokens != seen {
		t.Errorf("usage reports %d completion tokens but %d were streamed", resp.Usage.CompletionTokens, seen)
	}
	if strings.TrimSpace(streamed.String()) != strings.TrimSpace(resp.Text) {
		t.Errorf("streamed text and the final response disagree:\n  streamed: %q\n  response: %q", streamed.String(), resp.Text)
	}
}

func TestContextCancelledBeforeInferIsAnError(t *testing.T) {
	// Nothing was generated, so there is no result to return -- this one really is
	// an error, unlike a mid-generation cancel.
	chat := openChat(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := chat.InferContext(ctx, deterministic("hi", 8)); err != context.Canceled {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestInferContextWithoutAStreamCallbackStillCancels(t *testing.T) {
	chat := openChat(t)
	ctx, cancel := context.WithTimeout(context.Background(), 150e6) // 150ms
	defer cancel()

	resp, err := chat.InferContext(ctx, deterministic("Count slowly from one to fifty in words, one per line.", 300))
	if err != nil {
		t.Fatalf("InferContext: %v", err)
	}
	if !resp.Cancelled() {
		t.Errorf("finish_reason = %q, want a timed-out generation to be cancelled", resp.FinishReason)
	}
}

func TestJSONSchemaOutputParses(t *testing.T) {
	// Assert on parsed structure, never a substring: upstream's schema grammar
	// PERMITS a ```json fence, and a fence is exactly what strings.Contains cannot
	// see. The core strips it; this is the test that says so.
	chat := openChat(t)

	req := deterministic("Describe Paris.", 120)
	req.JSONSchema = map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city":    map[string]any{"type": "string"},
			"country": map[string]any{"type": "string"},
		},
		"required":             []string{"city", "country"},
		"additionalProperties": false,
	}

	resp, err := chat.Infer(req)
	if err != nil {
		t.Fatalf("Infer with a schema: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal([]byte(resp.Text), &got); err != nil {
		t.Fatalf("schema-constrained output does not parse as JSON: %v\n  got: %q", err, resp.Text)
	}
	for _, key := range []string{"city", "country"} {
		v, ok := got[key]
		if !ok {
			t.Errorf("required property %q is missing from %v", key, got)
			continue
		}
		if _, ok := v.(string); !ok {
			t.Errorf("property %q is %T, want a string per the schema", key, v)
		}
	}
	if len(got) != 2 {
		t.Errorf("additionalProperties:false was not honoured: %v", got)
	}
}

func TestRawGrammarConstrainsGeneration(t *testing.T) {
	chat := openChat(t)

	req := deterministic("Pick a colour.", 16)
	req.Grammar = `root ::= "red" | "blue"`

	resp, err := chat.Infer(req)
	if err != nil {
		t.Fatalf("Infer with a grammar: %v", err)
	}
	if got := strings.TrimSpace(resp.Text); got != "red" && got != "blue" {
		t.Errorf("grammar did not constrain generation: %q", resp.Text)
	}
}

func TestSchemaAndGrammarTogetherIsAnError(t *testing.T) {
	// A precedence rule here would be a silent winner between two output
	// constraints. The core rejects it, and the binding does NOT pre-empt that
	// locally -- the error must come from the core, in the same code every binding
	// reports.
	chat := openChat(t)

	req := deterministic("hi", 16)
	req.JSONSchema = map[string]any{"type": "object"}
	req.Grammar = `root ::= "x"`

	_, err := chat.Infer(req)
	if err == nil {
		t.Fatal("expected an error for json_schema + grammar")
	}
	var me *modelnexus.Error
	if !errorsAs(err, &me) {
		t.Fatalf("error is %T, want *modelnexus.Error", err)
	}
	if me.Code != "INVALID_REQUEST" {
		t.Errorf("code = %q, want INVALID_REQUEST", me.Code)
	}
}

func TestCacheStatusAndClear(t *testing.T) {
	// The assertion that matters is that the clear is OBSERVABLE. A clear that
	// silently did nothing would still return a well-formed CacheState, and the next
	// inference would still be correct -- just slow, and still holding the previous
	// tenant's conversation. So: infer, see a non-zero cache, clear, see zero, then
	// infer again and prove the handle still works.
	chat := openChat(t)
	req := deterministic("Name the capital of France in one word.", 16)

	if _, err := chat.Infer(req); err != nil {
		t.Fatalf("Infer: %v", err)
	}

	before, err := chat.CacheStatus()
	if err != nil {
		t.Fatalf("CacheStatus: %v", err)
	}
	if before.Tokens <= 0 {
		t.Fatalf("tokens = %d after an inference, want a non-empty cache", before.Tokens)
	}
	if before.NCtx <= 0 {
		t.Errorf("n_ctx = %d, want the engine's context window", before.NCtx)
	}

	// Status is the non-destructive call -- the binding's stand-in for the ABI's
	// "a NULL request reads status, it does not clear". Asking twice must not empty
	// the cache; getting that backwards would make an innocent-looking call wipe a
	// conversation.
	again, err := chat.CacheStatus()
	if err != nil {
		t.Fatalf("CacheStatus (second): %v", err)
	}
	if again.Tokens != before.Tokens {
		t.Errorf("reading the status changed it: %d then %d", before.Tokens, again.Tokens)
	}

	cleared, err := chat.ClearCache()
	if err != nil {
		t.Fatalf("ClearCache: %v", err)
	}
	if cleared.Tokens != 0 {
		t.Errorf("tokens = %d after a clear, want 0", cleared.Tokens)
	}

	after, err := chat.CacheStatus()
	if err != nil {
		t.Fatalf("CacheStatus (after clear): %v", err)
	}
	if after.Tokens != 0 {
		t.Errorf("the clear did not persist: status reports %d tokens", after.Tokens)
	}

	resp, err := chat.Infer(req)
	if err != nil {
		t.Fatalf("Infer after ClearCache: %v", err)
	}
	if !strings.Contains(resp.Text, "Paris") {
		t.Errorf("the engine stopped working after a clear: %q", resp.Text)
	}
}

func TestCacheOnAClosedChatIsATypedError(t *testing.T) {
	chat, err := modelnexus.Open(model(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	chat.Close()

	for _, tc := range []struct {
		name string
		call func() (modelnexus.CacheState, error)
	}{
		{"CacheStatus", chat.CacheStatus},
		{"ClearCache", chat.ClearCache},
	} {
		_, err := tc.call()
		if err == nil {
			t.Fatalf("%s on a closed Chat returned no error", tc.name)
		}
		var me *modelnexus.Error
		if !errorsAs(err, &me) {
			t.Fatalf("%s error is %T, want *modelnexus.Error", tc.name, err)
		}
		if me.Code != "ENGINE_CLOSED" {
			t.Errorf("%s code = %q, want ENGINE_CLOSED", tc.name, me.Code)
		}
	}
}
