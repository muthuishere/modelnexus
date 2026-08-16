// streaming -- print tokens as they arrive, and stop early from the callback.
//
// Stopping is the point. Before 0.2.0 a consumer who walked away — a closed stream, a
// user pressing stop — could not tell the model, so it generated to completion and you
// paid for all of it. Returning false from onToken ends the turn now.
//
//	MODELNEXUS_MODEL=/path/to/model.gguf go run ./streaming
package main

import (
	"fmt"
	"os"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

func main() {
	path := os.Getenv("MODELNEXUS_MODEL")
	if path == "" {
		fmt.Fprintln(os.Stderr, "MODELNEXUS_MODEL is not set — point it at a tool-capable GGUF")
		os.Exit(1)
	}

	// Quieten the engine so the streamed tokens are the only thing on the terminal.
	if err := modelnexus.SetLogLevel(modelnexus.LogError); err != nil {
		fmt.Fprintln(os.Stderr, "log level:", err)
		os.Exit(1)
	}

	chat, err := modelnexus.Open(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer chat.Close()

	temp := 0.0
	seed := uint32(42)
	maxTokens := 200 // deliberately more than we intend to read

	const budget = 20

	fmt.Print("streaming: ")
	seen := 0
	resp, err := chat.InferStream(modelnexus.Request{
		Messages: []modelnexus.Message{
			{Role: "user", Content: "List the planets of the solar system, one per line."},
		},
		Temperature: &temp,
		Seed:        &seed,
		MaxTokens:   &maxTokens,
	}, func(piece string) bool {
		fmt.Print(piece)
		seen++
		// The callback answers "keep going?", not "stop?". Returning false here ends
		// generation before the next token is decoded — the model never produces the
		// remaining 180 tokens, so they cost nothing.
		return seen < budget
	})
	fmt.Println()

	if err != nil {
		fmt.Fprintln(os.Stderr, "infer:", err)
		os.Exit(1)
	}

	// A cancelled generation is a RESULT, not an error: the response is complete, the
	// text is what was really produced, and the usage counts are the tokens you were
	// really charged for. err is nil above precisely because nothing went wrong.
	fmt.Println()
	fmt.Println("finish_reason:", resp.FinishReason)
	fmt.Println("cancelled:    ", resp.Cancelled())
	fmt.Printf("usage:         %d prompt + %d completion (asked for up to %d)\n",
		resp.Usage.PromptTokens, resp.Usage.CompletionTokens, maxTokens)
	fmt.Printf("pieces seen:   %d, response text length: %d bytes\n", seen, len(resp.Text))

	if !resp.Cancelled() {
		fmt.Fprintln(os.Stderr, "expected the callback to have stopped generation")
		os.Exit(1)
	}
}
