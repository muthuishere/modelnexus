// lora -- apply a LoRA adapter to a live engine, then take it off again.
//
// Adapters load against the model already in memory: no reload, no second copy of the
// weights, and several can be active at once with independent scales. They change
// *behaviour* — tone, output format, tool-call reliability — not knowledge. For facts,
// retrieve.
//
// The adapter and the base model are a matched PAIR: an adapter is built for one
// architecture and one tensor layout, and will not load against an arbitrary GGUF.
// Hence two env vars rather than reusing MODELNEXUS_MODEL.
//
//	MODELNEXUS_LORA_BASE=/path/to/base.gguf MODELNEXUS_LORA=/path/to/adapter.gguf go run ./lora
package main

import (
	"fmt"
	"os"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

// The adapter used to develop this example removes the base model's refusal
// behaviour, so a prompt the base declines is the one place the difference is legible.
const prompt = "Say something rude about the weather in one sentence."

// Scale is a dial, not a switch. This adapter is f16 against a q4 base, and at 1.0 it
// overwhelms the model — output degenerates into fragments. 0.25 shifts behaviour and
// keeps the model coherent. Any adapter you did not train yourself deserves this sweep.
const scale = 0.25

func main() {
	base := os.Getenv("MODELNEXUS_LORA_BASE")
	adapter := os.Getenv("MODELNEXUS_LORA")
	if base == "" || adapter == "" {
		fmt.Fprintln(os.Stderr, "set MODELNEXUS_LORA_BASE and MODELNEXUS_LORA to a matched base/adapter pair")
		os.Exit(1)
	}

	if err := modelnexus.SetLogLevel(modelnexus.LogError); err != nil {
		fmt.Fprintln(os.Stderr, "log level:", err)
		os.Exit(1)
	}

	chat, err := modelnexus.Open(base)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer chat.Close()

	// Temperature 0 and a fixed seed, so the only thing that can move the output between
	// the three calls below is the adapter.
	temp := 0.0
	seed := uint32(3)
	maxTokens := 60
	ask := func() string {
		resp, err := chat.Infer(modelnexus.Request{
			Messages:    []modelnexus.Message{{Role: "user", Content: prompt}},
			Temperature: &temp,
			Seed:        &seed,
			MaxTokens:   &maxTokens,
			// Each call must be provably independent, or the previous call's KV prefix
			// — computed under a different adapter set — could be reused underneath it.
			ReuseCache: modelnexus.ReuseCacheOff(),
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "infer:", err)
			os.Exit(1)
		}
		return resp.Text
	}

	fmt.Println("prompt:", prompt)
	fmt.Println()

	before := ask()
	fmt.Println("--- base model ---")
	fmt.Println(before)
	fmt.Println()

	id, err := chat.LoadLoRA(adapter, scale)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load lora:", err)
		os.Exit(1)
	}
	applied, err := chat.LoRAs()
	if err != nil {
		fmt.Fprintln(os.Stderr, "list loras:", err)
		os.Exit(1)
	}
	fmt.Printf("--- adapter %d applied at scale %.2f ---\n", id, applied[0].Scale)
	during := ask()
	fmt.Println(during)
	fmt.Println()

	// ClearLoRAs unloads every adapter and reapplies nothing, so the engine is back to
	// the weights it loaded from disk.
	if err := chat.ClearLoRAs(); err != nil {
		fmt.Fprintln(os.Stderr, "clear loras:", err)
		os.Exit(1)
	}
	after := ask()
	fmt.Println("--- adapter cleared ---")
	fmt.Println(after)
	fmt.Println()

	if before != after {
		// This is the check worth failing on: removing an adapter must restore the base
		// model exactly, or the engine is carrying state it should not.
		fmt.Fprintln(os.Stderr, "clearing the adapter did not restore the base model's output")
		os.Exit(1)
	}
	fmt.Println("clearing restored the base output byte for byte.")
	if before == during {
		fmt.Println("this adapter did not change the answer to this particular prompt.")
	} else {
		fmt.Println("the adapter changed the answer; the base model was restored by clearing it.")
	}
}
