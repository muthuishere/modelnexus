// counting -- how big is this conversation, before you commit to sending it?
//
// CountTokens applies the model's chat template and tokenizes. It creates no context,
// decodes nothing and does not touch the KV cache, so it is safe between inferences.
// It lives in the ABI because counting needs the model's vocabulary AND its parsed
// chat template, and no binding holds either — a tokenizer bolted on in Go would be a
// different tokenizer.
//
//	MODELNEXUS_MODEL=/path/to/model.gguf go run ./counting
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

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

	// A deliberately small window, so the budget is something you can watch fill up.
	chat, err := modelnexus.Open(path, modelnexus.WithContextSize(2048))
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer chat.Close()

	messages := []modelnexus.Message{
		{Role: "system", Content: "You are a support agent for a bicycle shop."},
		{Role: "user", Content: "My rear derailleur skips under load in the two lowest gears."},
	}

	// Grow the conversation and watch the count against the window. This is the loop a
	// real agent runs before every call, to decide whether to trim history.
	fmt.Println("messages   tokens   n_ctx   used")
	for i := 0; i < 5; i++ {
		count, err := chat.CountTokens(modelnexus.Request{Messages: messages})
		if err != nil {
			fmt.Fprintln(os.Stderr, "count:", err)
			os.Exit(1)
		}
		fmt.Printf("%8d %8d %7d %5.1f%%\n",
			len(messages), count.Tokens, count.NCtx,
			100*float64(count.Tokens)/float64(count.NCtx))

		messages = append(messages,
			modelnexus.Message{Role: "assistant", Content: strings.Repeat(
				"Check the cable tension at the barrel adjuster and index the shifter again. ", 6)},
			modelnexus.Message{Role: "user", Content: "That did not fix it. What else?"})
	}

	// Tools are part of the prompt too — they are rendered by the chat template — so
	// counting without them under-reports a tool-calling request.
	//
	// The parameter schema is a json.RawMessage rather than a map: Go marshals map keys
	// in sorted order, which changes the rendered prompt and therefore the count. It is
	// one token here; it is the reason the number differs between languages if you let
	// it happen.
	tools := []modelnexus.Tool{{
		Type: "function",
		Function: modelnexus.ToolFunction{
			Name:        "lookup_part",
			Description: "Find a spare part by model and component name",
			Parameters: json.RawMessage(`{
			  "type": "object",
			  "properties": {
			    "model":     {"type": "string"},
			    "component": {"type": "string"}
			  },
			  "required": ["model", "component"]
			}`),
		},
	}}

	bare, err := chat.CountTokens(modelnexus.Request{Messages: messages[:2]})
	if err != nil {
		fmt.Fprintln(os.Stderr, "count:", err)
		os.Exit(1)
	}
	withTools, err := chat.CountTokens(modelnexus.Request{Messages: messages[:2], Tools: tools})
	if err != nil {
		fmt.Fprintln(os.Stderr, "count:", err)
		os.Exit(1)
	}
	fmt.Println()
	fmt.Printf("the same two messages cost %d tokens, or %d once one tool declaration is attached (+%d)\n",
		bare.Tokens, withTools.Tokens, withTools.Tokens-bare.Tokens)
}
