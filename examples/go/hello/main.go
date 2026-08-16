// hello -- load a GGUF, run one inference, print the answer.
//
// The smallest thing that works. There is no server to start, no port to pick and no
// subprocess: Open maps the model into this process and Infer decodes in it.
//
//	MODELNEXUS_MODEL=/path/to/model.gguf go run ./hello
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

	// llama.cpp narrates the load. The bridge already owns that sink and defaults to
	// WARN; dropping to ERROR leaves this program's own output as the only output.
	// Set it before Open — logging starts during the load, so afterwards is too late.
	if err := modelnexus.SetLogLevel(modelnexus.LogError); err != nil {
		fmt.Fprintln(os.Stderr, "log level:", err)
		os.Exit(1)
	}

	// Open loads the weights and builds the inference context. It rejects a model whose
	// chat template cannot do tool calling, rather than loading it and degrading tool
	// calls to prose later — the failure arrives here, where it is cheap to read.
	chat, err := modelnexus.Open(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer chat.Close()

	// Generation parameters are pointers so that "unset" and "zero" stay distinct: a
	// Temperature of 0 is a legitimate request for greedy decoding, and a plain float64
	// could not tell it apart from a caller who never set the field.
	temp := 0.0
	seed := uint32(42)
	maxTokens := 64

	resp, err := chat.Infer(modelnexus.Request{
		Messages: []modelnexus.Message{
			{Role: "user", Content: "Name the capital of France. Answer in one word."},
		},
		Temperature: &temp,
		Seed:        &seed,
		MaxTokens:   &maxTokens,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "infer:", err)
		os.Exit(1)
	}

	fmt.Println("answer:", resp.Text)
	fmt.Printf("tokens: %d prompt + %d completion, finish_reason=%s\n",
		resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.FinishReason)
}
