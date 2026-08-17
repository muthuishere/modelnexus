// A toolnexus agent whose LLM is modelnexus: real tool calling, no network.
//
// toolnexus's in-process seam is "supply one thing: a method that answers a
// request" (cookbook/local-and-in-process-models). In Go that is an
// http.RoundTripper. Everything else -- the loop, MCP servers, skills -- is
// untouched, which is the point: nothing here teaches toolnexus about
// modelnexus, and neither library depends on the other (modelnexus ADR-0003).
//
// The two meet at the OpenAI wire shape, and nowhere else. modelnexus is built
// over llama.cpp's common_chat, so it already EMITS OpenAI-shaped tool_calls --
// this adapter is a translation, not an implementation.
//
//	MODELNEXUS_MODEL=/path/model.gguf go run ./toolnexus
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
	tn "github.com/muthuishere/toolnexus/golang"
)

// localRoundTripper answers toolnexus's LLM call from a model in this process.
// No socket is opened; BaseURL below is deliberately unreachable to prove it.
type localRoundTripper struct{ chat *modelnexus.Chat }

func (rt *localRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	// The wire shape and modelnexus's Message are NOT the same for tool calls, and
	// unmarshalling straight into modelnexus.Message loses them SILENTLY: OpenAI
	// nests {id, type, function:{name, arguments}}, modelnexus flattens to
	// {id, name, arguments}. json.Unmarshal produces entries with every field
	// empty and no error, and the failure surfaces one turn later as
	// "Missing tool call type" -- far from the cause. Translate explicitly.
	var body struct {
		Messages []struct {
			Role       string `json:"role"`
			Content    string `json:"content"`
			ToolCallID string `json:"tool_call_id"`
			ToolCalls  []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
		Tools []modelnexus.Tool `json:"tools"`
	}
	raw, _ := io.ReadAll(req.Body)
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, err
	}

	msgs := make([]modelnexus.Message, 0, len(body.Messages))
	for _, m := range body.Messages {
		mm := modelnexus.Message{Role: m.Role, Content: m.Content, ToolCallID: m.ToolCallID}
		for _, c := range m.ToolCalls {
			mm.ToolCalls = append(mm.ToolCalls, modelnexus.ToolCall{
				ID: c.ID, Name: c.Function.Name, Arguments: c.Function.Arguments,
			})
		}
		msgs = append(msgs, mm)
	}

	temp := 0.0
	seed := uint32(7)
	maxTok := 256
	res, err := rt.chat.Infer(modelnexus.Request{
		Messages:    msgs,
		Tools:       body.Tools,
		Temperature: &temp,
		Seed:        &seed,
		MaxTokens:   &maxTok,
	})
	if err != nil {
		return nil, err
	}

	// One OpenAI choice. content OR tool_calls, never both -- toolnexus branches
	// on which is present, and sending both is how an adapter makes a loop run
	// twice for one turn.
	msg := map[string]any{"role": "assistant"}
	if len(res.ToolCalls) > 0 {
		calls := make([]any, 0, len(res.ToolCalls))
		for _, c := range res.ToolCalls {
			calls = append(calls, map[string]any{
				"id": c.ID, "type": "function",
				"function": map[string]any{"name": c.Name, "arguments": c.Arguments},
			})
		}
		msg["tool_calls"] = calls
	} else {
		msg["content"] = res.Text
	}

	out, _ := json.Marshal(map[string]any{
		"choices": []any{map[string]any{"message": msg, "finish_reason": res.FinishReason}},
		"usage": map[string]any{
			"prompt_tokens": res.Usage.PromptTokens, "completion_tokens": res.Usage.CompletionTokens,
			"total_tokens": res.Usage.TotalTokens,
		},
	})
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(bytes.NewReader(out)),
		Request:    req,
	}, nil
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

	agent := tn.CreateClient(tn.ClientOptions{
		BaseURL:    "http://127.0.0.1:1/v1", // dead on purpose: nothing may reach it
		Style:      tn.StyleOpenAI,
		Model:      "modelnexus-local",
		APIKey:     "unused",
		HTTPClient: &http.Client{Transport: &localRoundTripper{chat: chat}},
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
	fmt.Println("\nno socket was opened: BaseURL points at a dead port.")
}
