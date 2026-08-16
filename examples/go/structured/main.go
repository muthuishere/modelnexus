// structured -- pass a JSON Schema, get output that is guaranteed to parse.
//
// A schema is compiled into a grammar that constrains decoding, so the model cannot
// emit a token that would break the shape. The usual small-model failure — your JSON
// plus an apology, or a truncated object — becomes impossible rather than unlikely.
//
//	MODELNEXUS_MODEL=/path/to/model.gguf go run ./structured
package main

import (
	"encoding/json"
	"fmt"
	"os"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

// Review is the shape we want back. Decoding straight into a struct is the point of
// the exercise: if this Unmarshal can fail, the feature has not delivered anything.
type Review struct {
	Sentiment string   `json:"sentiment"`
	Rating    int      `json:"rating"`
	Topics    []string `json:"topics"`
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

	chat, err := modelnexus.Open(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer chat.Close()

	// Any value that marshals to a JSON Schema object works. "required" and "enum" are
	// worth setting: they are constraints the grammar can enforce, which makes them
	// free, unlike a prompt instruction the model may ignore.
	//
	// A json.RawMessage rather than a map[string]any, because PROPERTY ORDER IS LOAD
	// BEARING here: the grammar makes the model emit the fields in the order the schema
	// lists them, so it decides "sentiment" before "rating" or the other way round. Go
	// marshals a map with its keys SORTED, which silently reorders the schema and puts
	// this example on a different decode path from the Python and JS ones.
	schema := json.RawMessage(`{
	  "type": "object",
	  "properties": {
	    "sentiment": {"type": "string", "enum": ["positive", "negative", "mixed"]},
	    "rating":    {"type": "integer", "minimum": 1, "maximum": 5},
	    "topics":    {"type": "array", "items": {"type": "string"}}
	  },
	  "required": ["sentiment", "rating", "topics"],
	  "additionalProperties": false
	}`)

	temp := 0.0
	seed := uint32(7)
	maxTokens := 120

	resp, err := chat.Infer(modelnexus.Request{
		Messages: []modelnexus.Message{{
			Role: "user",
			Content: "Classify this review: \"The battery lasts two days and the screen is " +
				"gorgeous, but the camera is mediocre and it costs too much.\"",
		}},
		JSONSchema:  schema,
		Temperature: &temp,
		Seed:        &seed,
		MaxTokens:   &maxTokens,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "infer:", err)
		os.Exit(1)
	}

	// No repair pass, no fence stripping, no retry loop. The core already removed the
	// ```json fence llama.cpp's generated grammar permits, so what arrives is JSON.
	var review Review
	if err := json.Unmarshal([]byte(resp.Text), &review); err != nil {
		fmt.Fprintln(os.Stderr, "the schema did not hold:", err)
		fmt.Fprintln(os.Stderr, "raw:", resp.Text)
		os.Exit(1)
	}

	fmt.Printf("sentiment: %s\n", review.Sentiment)
	fmt.Printf("rating:    %d\n", review.Rating)
	fmt.Printf("topics:    %v\n", review.Topics)
	fmt.Printf("(parsed from %d bytes of model output, finish_reason=%s)\n",
		len(resp.Text), resp.FinishReason)
}
