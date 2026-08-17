// A toolnexus agent whose LLM is modelnexus: real tool calling, no network.
//
// toolnexus 0.16.0 added CreateInProcessClient, so the seam is now a function
// that answers a request — not an http.RoundTripper.
//
// Measured, not asserted: 142 lines of code became 116, and `bytes`, `io` and
// `net/http` left the imports entirely. What went is the HTTP envelope — no
// Response to build, no status code, no header map, no `choices` wrapper — and
// the hand-written translation into and out of the nested
// `function:{name,arguments}` form, since both libraries now speak flat tool
// calls. What arrived is one small decode helper.
//
// 18% is a fair number for a file this size, and the shape of what is left
// matters more than the count: everything below is about the MODEL. There is
// no longer any code here about HTTP.
//
// Nothing here teaches toolnexus about modelnexus, and neither library depends
// on the other (modelnexus ADR-0003). They meet at one shape: messages in, one
// assistant message out.
//
//	MODELNEXUS_MODEL=/path/model.gguf go run ./toolnexus
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
	tn "github.com/muthuishere/toolnexus/golang"
)

// generate is the entire adapter. It receives the assembled request and returns
// one assistant message — content to finish, tool calls to ask for tools.
//
// Both libraries speak flat tool calls ({id, name, arguments}), so there is
// nothing to translate in that direction any more. The nested wire form is
// toolnexus's problem now, which is where it belongs.
func generate(chat *modelnexus.Chat) func(tn.InProcessRequest) (tn.InProcessResponse, error) {
	return func(req tn.InProcessRequest) (tn.InProcessResponse, error) {
		msgs, err := decode[[]modelnexus.Message](req.Messages)
		if err != nil {
			return tn.InProcessResponse{}, fmt.Errorf("messages: %w", err)
		}
		tools, err := decode[[]modelnexus.Tool](req.Tools)
		if err != nil {
			return tn.InProcessResponse{}, fmt.Errorf("tools: %w", err)
		}

		temp, seed, maxTok := 0.0, uint32(7), 256
		res, err := chat.Infer(modelnexus.Request{
			Messages:    msgs,
			Tools:       tools,
			Temperature: &temp,
			Seed:        &seed,
			MaxTokens:   &maxTok,
		})
		if err != nil {
			return tn.InProcessResponse{}, err
		}

		out := tn.InProcessResponse{
			Usage: &tn.InProcessUsage{
				PromptTokens:     res.Usage.PromptTokens,
				CompletionTokens: res.Usage.CompletionTokens,
				TotalTokens:      res.Usage.TotalTokens,
			},
		}
		// Content OR tool calls, never both — toolnexus branches on which is set,
		// and supplying both is how an adapter makes one turn run twice.
		if len(res.ToolCalls) > 0 {
			for _, c := range res.ToolCalls {
				// Arguments stays a json.RawMessage: it is already encoded, and
				// toolnexus passes a pre-encoded value through untouched. Decoding
				// it here only to have it re-encoded would be a round trip that can
				// only lose things.
				out.ToolCalls = append(out.ToolCalls, tn.InProcessToolCall{
					ID: c.ID, Name: c.Name, Arguments: c.Arguments,
				})
			}
			return out, nil
		}
		out.Content = res.Text
		return out, nil
	}
}

// decode re-reads toolnexus's []any into modelnexus's own types. The messages
// arrive as decoded JSON rather than as a typed struct, because the request is
// assembled from many sources; a round trip through json is the honest way to
// land it in another library's types without either one importing the other.
func decode[T any](in []any) (T, error) {
	var out T
	if len(in) == 0 {
		return out, nil
	}
	raw, err := json.Marshal(in)
	if err != nil {
		return out, err
	}
	return out, json.Unmarshal(raw, &out)
}

func main() {
	model := os.Getenv("MODELNEXUS_MODEL")
	if model == "" {
		fmt.Fprintln(os.Stderr, "MODELNEXUS_MODEL is not set — point it at a tool-capable GGUF")
		os.Exit(1)
	}
	_ = modelnexus.SetLogLevel(modelnexus.LogError)

	chat, err := modelnexus.Open(model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer chat.Close()

	called := 0
	tk, err := tn.CreateToolkit(context.Background(), tn.Options{
		Builtins: false,
		ExtraTools: []tn.Tool{{
			Name:        "get_weather",
			Description: "Current weather for a city.",
			InputSchema: tn.JSONSchema{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
				"required":   []string{"city"},
			},
			Source: tn.ToolSource("native"),
			Execute: func(args map[string]any, _ *tn.ToolContext) (tn.ToolResult, error) {
				called++
				fmt.Printf("  [tool ran] get_weather(%v)\n", args["city"])
				return tn.ToolResult{Output: fmt.Sprintf("18C and raining in %v", args["city"])}, nil
			},
		}},
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "toolkit:", err)
		os.Exit(1)
	}

	// No BaseURL, no APIKey, no Style. There is no wire to describe, and the
	// previous version of this file had to invent all three — a dead port, an
	// unused key, and a style for a protocol nobody speaks.
	agent := tn.CreateInProcessClient(tn.InProcessOptions{
		Model:    "modelnexus-local",
		Generate: generate(chat),
	})

	fmt.Println("asking an agent with one tool, model running in this process...")
	res, err := agent.Run(context.Background(), "What is the weather in Paris? Use the tool.", tk)
	if err != nil {
		fmt.Fprintln(os.Stderr, "run:", err)
		os.Exit(1)
	}

	fmt.Println()
	fmt.Println("answer:      ", res.Text)
	fmt.Println("tools called:", called)
	if called == 0 {
		fmt.Fprintln(os.Stderr, "\nthe model never called the tool — a small model may need a clearer prompt,")
		fmt.Fprintln(os.Stderr, "but the wiring above is what matters and it ran.")
		os.Exit(1)
	}
	fmt.Println("\nno socket was opened, and no URL was configured to prove it with.")
}
