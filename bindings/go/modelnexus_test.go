package modelnexus_test

import (
	"fmt"
	"os"
	"strings"
	"testing"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

// TestMain says loudly when the suite is about to skip nearly everything.
//
// A skipped suite must not look like a passing one: `go test` prints "ok" either way,
// and every real assertion in here needs a model. Run the suite with -v -- go test
// discards a passing package's output otherwise, so -v is the only invocation in
// which any banner, from here or from t.Skip, can be seen at all.
func TestMain(m *testing.M) {
	if os.Getenv("MODELNEXUS_MODEL") == "" {
		bar := strings.Repeat("=", 72)
		fmt.Fprintf(os.Stderr, "%s\nSKIPPING the modelnexus inference suite: MODELNEXUS_MODEL is not set.\n"+
			"Nothing below tested a model. Set MODELNEXUS_MODEL to a tool-capable GGUF\n"+
			"to run the real conformance checks.\n%s\n", bar, bar)
	}
	os.Exit(m.Run())
}

// model returns the GGUF to test against, or skips.
//
// Skipping rather than failing is deliberate: the binding's own correctness is
// worth testing on a machine that has no multi-gigabyte model sitting around, and
// a suite that cannot run at all teaches nobody anything.
func model(t *testing.T) string {
	t.Helper()
	path := os.Getenv("MODELNEXUS_MODEL")
	if path == "" {
		t.Skip("set MODELNEXUS_MODEL to a tool-capable GGUF to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("MODELNEXUS_MODEL=%s is not readable: %v", path, err)
	}
	return path
}

func TestVersion(t *testing.T) {
	v, err := modelnexus.Version()
	if err != nil {
		t.Fatalf("Version: %v", err)
	}
	if !strings.Contains(v, "llamabridge") {
		t.Errorf("version %q does not identify the bridge", v)
	}
	if !strings.Contains(v, "llama.cpp") {
		t.Errorf("version %q does not name the linked llama.cpp tag", v)
	}
}

func TestInfoReportsToolCapability(t *testing.T) {
	info, err := modelnexus.Info(model(t))
	if err != nil {
		t.Fatalf("Info: %v", err)
	}
	if !info.SupportsTools {
		t.Error("expected a tool-capable model; the core rejects non-tool models at load")
	}
	if info.ChatFormat == "" {
		t.Error("expected a chat format to be reported")
	}
}

func TestInfer(t *testing.T) {
	chat, err := modelnexus.Open(model(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer chat.Close()

	max, seed := 16, uint32(1)
	resp, err := chat.Infer(modelnexus.Request{
		Messages:  []modelnexus.Message{{Role: "user", Content: "Reply with exactly: pong"}},
		MaxTokens: &max,
		Seed:      &seed,
	})
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}
	if resp.Type != "assistant_text" {
		t.Errorf("type = %q, want assistant_text", resp.Type)
	}
	if resp.Text == "" {
		t.Error("expected non-empty text")
	}
	if resp.Usage.TotalTokens == 0 {
		t.Error("expected token usage to be reported")
	}
}

func TestInferStreamDeliversTokensAndFinalResponse(t *testing.T) {
	chat, err := modelnexus.Open(model(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer chat.Close()

	var pieces []string
	max, seed := 24, uint32(1)
	resp, err := chat.InferStream(modelnexus.Request{
		Messages:  []modelnexus.Message{{Role: "user", Content: "Count: 1 2 3"}},
		MaxTokens: &max,
		Seed:      &seed,
	}, func(p string) bool { pieces = append(pieces, p); return true })
	if err != nil {
		t.Fatalf("InferStream: %v", err)
	}
	if len(pieces) == 0 {
		t.Fatal("expected streamed token pieces")
	}
	// Streaming must not be a different code path with a different result: the
	// final response carries the same accounting as the non-streaming call.
	if resp.Usage.CompletionTokens == 0 {
		t.Error("expected usage on the final streamed response")
	}
}

func TestEventHandlerSurvivesTheCall(t *testing.T) {
	// The core stores the event callback on the engine and keeps calling it, so a
	// binding that lets the trampoline die after Open crashes the process later.
	// This test exists because that is exactly what happened during development.
	var events []string
	chat, err := modelnexus.Open(model(t), modelnexus.WithEventHandler(func(e string) {
		events = append(events, e)
	}))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer chat.Close()

	max := 8
	if _, err := chat.Infer(modelnexus.Request{
		Messages:  []modelnexus.Message{{Role: "user", Content: "hi"}},
		MaxTokens: &max,
	}); err != nil {
		t.Fatalf("Infer after Open: %v", err)
	}
	if len(events) == 0 {
		t.Error("expected at least one lifecycle event")
	}
}

func TestMissingModelIsATypedError(t *testing.T) {
	_, err := modelnexus.Open("/definitely/not/here.gguf")
	if err == nil {
		t.Fatal("expected an error for a missing model")
	}
	var me *modelnexus.Error
	if !errorsAs(err, &me) {
		t.Fatalf("error is %T, want *modelnexus.Error", err)
	}
	// The code is part of the cross-binding contract: Python reports the same one.
	if me.Code != "MODEL_LOAD_FAILED" {
		t.Errorf("code = %q, want MODEL_LOAD_FAILED", me.Code)
	}
}

func TestUseAfterCloseIsRejected(t *testing.T) {
	chat, err := modelnexus.Open(model(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	chat.Close()
	chat.Close() // idempotent

	if _, err := chat.Infer(modelnexus.Request{
		Messages: []modelnexus.Message{{Role: "user", Content: "hi"}},
	}); err == nil {
		t.Fatal("expected an error when inferring on a closed Chat")
	}
}

// errorsAs is errors.As, kept local so the test file has no extra import noise.
func errorsAs(err error, target **modelnexus.Error) bool {
	for err != nil {
		if e, ok := err.(*modelnexus.Error); ok {
			*target = e
			return true
		}
		u, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}
