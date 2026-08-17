// Does the PUBLISHED Windows native actually load and run, on real Windows?
//
// natives.yml going green proves the Windows bridge COMPILED. Nothing has ever
// proved it LOADS. Those are different claims, and 0.2.0 shipped a
// windows-x86_64 asset on the strength of the weaker one.
//
// Go rather than Python, for two reasons: a bare Windows box has no Python, and
// this way the run also exercises the GO BINDING on Windows -- which has never
// been done either. It uses Fetch(), so it walks the exact path a user walks.
//
// Build (from the repo root, on any machine):
//
//	GOOS=windows GOARCH=amd64 go build -o winverify.exe ./core/tests/winverify
//
// GOARCH=amd64 on purpose: the published native is windows-x86_64, so on an ARM
// Windows box both the binary and the DLLs run under x64 emulation together. A
// GOARCH=arm64 build could not load an x64 DLL at all.
//
// Usage:  winverify.exe [path\to\model.gguf]
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

var checks, failures int

func ok(cond bool, what string, detail ...string) {
	checks++
	d := ""
	if len(detail) > 0 && detail[0] != "" {
		d = "  [" + strings.TrimSpace(detail[0]) + "]"
	}
	if cond {
		fmt.Printf("  ok    %s%s\n", what, d)
	} else {
		fmt.Printf("  FAIL  %s%s\n", what, d)
		failures++
	}
}

func main() {
	fmt.Println("modelnexus Windows verification")
	fmt.Printf("  GOOS/GOARCH of this binary: windows/%s\n", archOf())

	// The real user path: Fetch pulls the published native into the cache, and
	// Open must then find it with no MODELNEXUS_LIB set. That combination is
	// what was broken on macOS until yesterday.
	if os.Getenv("MODELNEXUS_LIB") == "" {
		dir, err := modelnexus.Fetch()
		if err != nil {
			fmt.Println("  FAIL  Fetch():", err)
			os.Exit(1)
		}
		fmt.Println("  fetched natives ->", dir)
	} else {
		fmt.Println("  using MODELNEXUS_LIB =", os.Getenv("MODELNEXUS_LIB"))
	}

	version, err := modelnexus.Version()
	if err != nil {
		// On Windows this is where a missing sibling DLL shows up: the bridge
		// resolves llama/ggml from its own directory, so a staging bug that
		// dropped one surfaces here and nowhere earlier.
		fmt.Println("  FAIL  the native library did not load:", err)
		os.Exit(1)
	}
	ok(true, "the native library loads", version)
	ok(strings.Contains(version, "0.2.0"), "it is the 0.2.0 bridge, not a stale asset", version)

	model := ""
	if len(os.Args) > 1 {
		model = os.Args[1]
	}
	if model == "" {
		fmt.Println()
		fmt.Println("  SKIP  inference — no model path given.")
		fmt.Println("        Half the check ran: the library loads and reports its version,")
		fmt.Println("        but nothing has generated a token on this machine.")
		fmt.Printf("\n%d checks, %d failures (inference SKIPPED)\n", checks, failures)
		os.Exit(1)
	}
	if _, err := os.Stat(model); err != nil {
		fmt.Println("  FAIL  model not found:", model)
		os.Exit(1)
	}

	_ = modelnexus.SetLogLevel(modelnexus.LogError)

	chat, err := modelnexus.Open(model, modelnexus.WithContextSize(2048))
	if err != nil {
		fmt.Println("  FAIL  Open:", err)
		os.Exit(1)
	}
	defer chat.Close()

	msgs := []modelnexus.Message{{Role: "user", Content: "Name the capital of France in one word."}}
	maxTok, seed, temp := 16, uint32(42), 0.0
	req := modelnexus.Request{Messages: msgs, MaxTokens: &maxTok, Seed: &seed, Temperature: &temp}

	// A 0.2.0-only entry point, so this cannot pass against a stale native.
	tc, err := chat.CountTokens(req)
	ok(err == nil && tc.Tokens > 0, "CountTokens works (0.2.0 entry point)", fmt.Sprint(tc.Tokens, " tokens"))
	ok(tc.NCtx == 2048, "the create config crossed the ABI", fmt.Sprint("n_ctx=", tc.NCtx))

	res, err := chat.Infer(req)
	ok(err == nil && strings.Contains(res.Text, "Paris"), "inference produces a correct answer", res.Text)

	// Cancellation: stop from the callback on the 3rd token.
	seen := 0
	long := modelnexus.Request{
		Messages: []modelnexus.Message{{Role: "user", Content: "Count from one to twenty in words."}},
	}
	n2, s2, t2 := 200, uint32(7), 0.0
	long.MaxTokens, long.Seed, long.Temperature = &n2, &s2, &t2
	sres, err := chat.InferStream(long, func(string) bool { seen++; return seen < 3 })
	ok(err == nil && sres.FinishReason == "cancelled", "cancellation works", sres.FinishReason)
	ok(seen == 3, "it stopped at the requested token", fmt.Sprint(seen, " tokens"))

	before, _ := chat.CacheStatus()
	cleared, err := chat.ClearCache()
	ok(err == nil && cleared.Tokens == 0, "ClearCache empties the cache", fmt.Sprint("was ", before.Tokens))

	again, err := chat.Infer(req)
	ok(err == nil && strings.Contains(again.Text, "Paris"), "the engine still works after a clear")

	schema := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"}},"required":["city","country"],"additionalProperties":false}`)
	jr := req
	jt := 120
	jr.MaxTokens = &jt
	jr.Messages = []modelnexus.Message{{Role: "user", Content: "Describe Paris."}}
	jr.JSONSchema = schema
	jres, err := chat.Infer(jr)
	var parsed map[string]any
	perr := json.Unmarshal([]byte(jres.Text), &parsed)
	ok(err == nil && perr == nil && len(parsed) == 2, "schema output parses and matches", jres.Text)

	fmt.Printf("\n%d checks, %d failures\n", checks, failures)
	if failures > 0 {
		os.Exit(1)
	}
}
