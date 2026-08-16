package modelnexus_test

import (
	"encoding/json"
	"strings"
	"testing"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

// TestJSONSchemaMapReordersProperties pins the trap documented on Request.JSONSchema.
//
// It needs no model: the damage is done by encoding/json before the ABI is reached.
// encoding/json sorts map keys, so a schema written as map[string]any arrives at the
// grammar with its properties alphabetised -- and under constrained decoding the model
// must emit fields in the order the grammar allows, so the reordering changes the
// answer, not just the layout. The comment explaining this is only believable while
// this test still fails when someone "simplifies" the fix away.
func TestJSONSchemaMapReordersProperties(t *testing.T) {
	// Author order is sentiment-then-rating; alphabetical order is the reverse, which
	// is what makes the two distinguishable at all.
	ordered := json.RawMessage(`{"type":"object","properties":{"sentiment":{"type":"string"},"rating":{"type":"integer"}},"required":["sentiment","rating"]}`)

	unordered := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"sentiment": map[string]any{"type": "string"},
			"rating":    map[string]any{"type": "integer"},
		},
		"required": []string{"sentiment", "rating"},
	}

	rawWire := marshalSchema(t, ordered)
	mapWire := marshalSchema(t, unordered)

	if !strings.Contains(rawWire, `"sentiment"`) || !strings.Contains(rawWire, `"rating"`) {
		t.Fatalf("schema did not survive marshalling: %s", rawWire)
	}

	if got := propertyOrder(rawWire); got != "sentiment,rating" {
		t.Errorf("json.RawMessage must preserve the author's property order, got %q in %s", got, rawWire)
	}
	if got := propertyOrder(mapWire); got != "rating,sentiment" {
		t.Errorf("map[string]any is expected to arrive alphabetised, got %q in %s", got, mapWire)
	}
	if rawWire == mapWire {
		t.Fatal("the two schemas serialised identically; the ordering trap this test pins has changed")
	}
}

func marshalSchema(t *testing.T, schema any) string {
	t.Helper()
	payload, err := json.Marshal(modelnexus.Request{
		Messages:   []modelnexus.Message{{Role: "user", Content: "rate this"}},
		JSONSchema: schema,
	})
	if err != nil {
		t.Fatalf("could not marshal request: %v", err)
	}
	return string(payload)
}

// propertyOrder reports the order the two property names appear on the wire.
func propertyOrder(wire string) string {
	sentiment := strings.Index(wire, `"sentiment":{`)
	rating := strings.Index(wire, `"rating":{`)
	if sentiment < 0 || rating < 0 {
		return ""
	}
	if sentiment < rating {
		return "sentiment,rating"
	}
	return "rating,sentiment"
}
