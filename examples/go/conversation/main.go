// conversation -- a multi-turn loop, with per-turn wall clock printed.
//
// Every turn resends the whole conversation, so the prompt grows without bound. The
// engine keeps what is already in its KV cache and re-decodes only the part that
// differs, which for an appending conversation is just the new turn. The cost of
// re-reading the prefix therefore stops growing.
//
// This example does not assert a speedup. It runs the SAME turns twice — once with
// reuse on (the default) and once with reuse_cache:false — and prints both clocks so
// the reader sees whatever this machine actually does.
//
//	MODELNEXUS_MODEL=/path/to/model.gguf go run ./conversation
package main

import (
	"fmt"
	"os"
	"time"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

// Scripted turns, so both runs do identical work and the only variable is the cache.
var turns = []string{
	"I am planning a week in Lisbon. Give me one neighbourhood to stay in.",
	"What is one dish I should eat there?",
	"Name one day trip within two hours.",
	"What is the weather like in October?",
	"One phrase of Portuguese I should learn?",
	"Is the metro worth using?",
	"One museum worth an afternoon?",
	"Sum up the trip in one sentence.",
}

// A system prompt long enough that the reused prefix is worth something from turn one.
const system = "You are a concise travel assistant. Answer in at most two short sentences. " +
	"Never list more than one option. Do not repeat the question back. Do not add caveats " +
	"about checking current information. Assume the traveller is an experienced adult who " +
	"has been to Europe before and wants opinions, not disclaimers."

func run(path string, reuse bool) ([]time.Duration, []int, error) {
	chat, err := modelnexus.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer chat.Close()

	messages := []modelnexus.Message{{Role: "system", Content: system}}
	temp := 0.0
	seed := uint32(11)
	maxTokens := 40

	var elapsed []time.Duration
	var prompts []int

	for _, q := range turns {
		messages = append(messages, modelnexus.Message{Role: "user", Content: q})

		req := modelnexus.Request{
			Messages:    messages,
			Temperature: &temp,
			Seed:        &seed,
			MaxTokens:   &maxTokens,
		}
		// ReuseCache is a pointer because the core's default is ON: a plain bool would
		// silently opt every caller out. Leaving it nil means "the core decides".
		if !reuse {
			req.ReuseCache = modelnexus.ReuseCacheOff()
		}

		start := time.Now()
		resp, err := chat.Infer(req)
		if err != nil {
			return nil, nil, err
		}
		elapsed = append(elapsed, time.Since(start))
		prompts = append(prompts, resp.Usage.PromptTokens)

		// Appending the reply is what makes the next prompt a strict extension of this
		// one — which is exactly the shape prefix reuse can exploit.
		messages = append(messages, modelnexus.Message{Role: "assistant", Content: resp.Text})
	}
	return elapsed, prompts, nil
}

func main() {
	path := os.Getenv("MODELNEXUS_MODEL")
	if path == "" {
		fmt.Fprintln(os.Stderr, "MODELNEXUS_MODEL is not set — point it at a tool-capable GGUF")
		os.Exit(1)
	}

	if err := modelnexus.SetLogLevel(modelnexus.LogError); err != nil {
		fmt.Fprintln(os.Stderr, "log level:", err)
		os.Exit(1)
	}

	// Each run gets a fresh engine so neither inherits the other's cache.
	reuseMS, prompts, err := run(path, true)
	if err != nil {
		fmt.Fprintln(os.Stderr, "reuse run:", err)
		os.Exit(1)
	}
	freshMS, _, err := run(path, false)
	if err != nil {
		fmt.Fprintln(os.Stderr, "no-reuse run:", err)
		os.Exit(1)
	}

	fmt.Println()
	fmt.Println("turn  prompt tokens   reuse on   reuse off")
	var totalReuse, totalFresh time.Duration
	for i := range turns {
		totalReuse += reuseMS[i]
		totalFresh += freshMS[i]
		fmt.Printf("%3d %12d %8.0f ms %8.0f ms\n",
			i+1, prompts[i],
			float64(reuseMS[i].Microseconds())/1000,
			float64(freshMS[i].Microseconds())/1000)
	}
	fmt.Printf("total %19.0f ms %8.0f ms\n",
		float64(totalReuse.Microseconds())/1000,
		float64(totalFresh.Microseconds())/1000)
	fmt.Println()

	// Wall clock, not a claim: these numbers are whatever this machine produced just
	// now. Each turn generates the same 40 tokens, so the difference between the
	// columns is the prefill the reuse run did not have to redo.
	fmt.Printf("prompt grew from %d to %d tokens across %d turns\n",
		prompts[0], prompts[len(prompts)-1], len(turns))
}
