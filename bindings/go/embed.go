package modelnexus

import (
	"encoding/json"
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ---------------------------------------------------------------- LoRA

// Adapter is one LoRA adapter applied to a Chat's context.
type Adapter struct {
	ID    int     `json:"id"`
	Path  string  `json:"path"`
	Scale float64 `json:"scale"`
}

type loraResult struct {
	Type     string    `json:"type"`
	ID       int       `json:"id"`
	Adapters []Adapter `json:"adapters"`
	Error    *Error    `json:"error"`
}

func (c *Chat) lora(op map[string]any) (*loraResult, error) {
	if c.handle == 0 {
		return nil, &Error{Code: "ENGINE_CLOSED", Message: "this Chat has already been closed"}
	}
	payload, err := json.Marshal(op)
	if err != nil {
		return nil, fmt.Errorf("modelnexus: could not encode LoRA request: %w", err)
	}
	raw := takeString(llbChatLora(c.handle, string(payload)))

	var res loraResult
	if err := json.Unmarshal([]byte(raw), &res); err != nil {
		return nil, fmt.Errorf("modelnexus: unparseable LoRA response: %w", err)
	}
	if res.Type == "error" {
		if res.Error != nil {
			return nil, res.Error
		}
		return nil, &Error{Code: "UNKNOWN", Message: "core reported an error with no detail"}
	}
	return &res, nil
}

// LoadLoRA loads a LoRA adapter and applies it, returning its id.
//
// Several adapters can be active at once; they apply in load order. Adapters change
// *behaviour* -- output format, tone, tool-call reliability -- not knowledge. For
// facts, retrieve.
func (c *Chat) LoadLoRA(path string, scale float64) (int, error) {
	res, err := c.lora(map[string]any{"op": "load", "path": path, "scale": scale})
	if err != nil {
		return 0, err
	}
	return res.ID, nil
}

// SetLoRAScale changes an adapter's scale. It takes effect on the next inference.
func (c *Chat) SetLoRAScale(id int, scale float64) error {
	_, err := c.lora(map[string]any{"op": "set", "id": id, "scale": scale})
	return err
}

// RemoveLoRA unloads one adapter and reapplies the rest.
func (c *Chat) RemoveLoRA(id int) error {
	_, err := c.lora(map[string]any{"op": "remove", "id": id})
	return err
}

// ClearLoRAs unloads every adapter, returning the model to its base behaviour.
func (c *Chat) ClearLoRAs() error {
	_, err := c.lora(map[string]any{"op": "clear"})
	return err
}

// LoRAs reports the adapters currently applied, in order.
func (c *Chat) LoRAs() ([]Adapter, error) {
	res, err := c.lora(map[string]any{"op": "list"})
	if err != nil {
		return nil, err
	}
	return res.Adapters, nil
}

// ---------------------------------------------------------------- embeddings

// Pooling is how token vectors are reduced to one vector per input.
type Pooling string

const (
	// PoolingMean averages token vectors -- the usual choice for sentence embeddings.
	PoolingMean Pooling = "mean"
	// PoolingCLS takes the first token's vector, used by BERT-style encoders.
	PoolingCLS Pooling = "cls"
	// PoolingLast takes the final token's vector, used by decoder-style embedders.
	PoolingLast Pooling = "last"
	// PoolingRank attaches the model's classification head. Required for Rerank,
	// and useless for anything else.
	PoolingRank Pooling = "rank"
	// PoolingNone returns no pooled vector at all.
	PoolingNone Pooling = "none"
)

// EmbedOptions configures OpenEmbedder.
type EmbedOptions struct {
	// Pooling strategy. Empty means the model's own default. Use PoolingRank for a
	// reranker -- Rerank refuses to run without it.
	Pooling Pooling
	// NCtx is the context size; 0 means the model default.
	NCtx int
	// NBatch caps how many tokens one input may have; 0 means the core's default
	// (llb_embed_create in core/include/llamabridge.h states it).
	NBatch int
	// OnEvent receives lifecycle events during load.
	OnEvent func(string)
}

// Embedder is a model loaded for embedding or reranking.
//
// Separate from Chat on purpose: embedding needs a context created with embeddings
// enabled and a pooling type fixed up front, and reranking needs PoolingRank
// specifically -- none of which can be switched on a generation context afterwards.
type Embedder struct {
	handle  uintptr
	eventID uintptr
	mu      sync.Mutex
	events  []string
}

// OpenEmbedder loads a model for embedding or reranking.
func OpenEmbedder(ggufPath string, opts *EmbedOptions) (*Embedder, error) {
	if err := ensureLoaded(); err != nil {
		return nil, err
	}
	initTrampolines()

	if opts == nil {
		opts = &EmbedOptions{}
	}
	config := map[string]any{}
	if opts.Pooling != "" {
		config["pooling"] = string(opts.Pooling)
	}
	if opts.NCtx > 0 {
		config["n_ctx"] = opts.NCtx
	}
	if opts.NBatch > 0 {
		config["n_batch"] = opts.NBatch
	}
	// "" is NULL to the ABI, and NULL means "every default, exactly as before the
	// parameter existed". An empty object is not the same request -- see Open.
	payload := ""
	if len(config) > 0 {
		encoded, err := json.Marshal(config)
		if err != nil {
			return nil, fmt.Errorf("modelnexus: could not encode config: %w", err)
		}
		payload = string(encoded)
	}

	e := &Embedder{}
	// Held for the life of the handle, like Chat's -- the core keeps the pointer.
	e.eventID = registerSink(eventSink, func(msg string) {
		e.mu.Lock()
		e.events = append(e.events, msg)
		e.mu.Unlock()
		if opts.OnEvent != nil {
			opts.OnEvent(msg)
		}
	})

	handle := llbEmbedCreate(ggufPath, payload, eventTrampoline, e.eventID)
	if handle == 0 {
		defer releaseSink(eventSink, e.eventID)
		e.mu.Lock()
		events := append([]string(nil), e.events...)
		e.mu.Unlock()
		detail := "unknown reason"
		if len(events) > 0 {
			detail = fmt.Sprint(events)
		}
		return nil, &Error{Code: "MODEL_LOAD_FAILED", Message: "could not load " + ggufPath + " (" + detail + ")"}
	}
	e.handle = handle
	return e, nil
}

type embedResponse struct {
	Type       string      `json:"type"`
	Dim        int         `json:"dim"`
	Embeddings [][]float32 `json:"embeddings"`
	Usage      Usage       `json:"usage"`
	Error      *Error      `json:"error"`
}

type rerankResponse struct {
	Type    string      `json:"type"`
	Results []RerankHit `json:"results"`
	Usage   Usage       `json:"usage"`
	Error   *Error      `json:"error"`
}

// RerankHit is one scored document.
//
// Index is the document's position in the ORIGINAL slice, because results come back
// reordered. Score is a raw model logit: comparable within one call, not across
// models, and not a probability.
type RerankHit struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
}

// Embed returns one vector per input, in input order.
//
// Vectors are L2-normalized, which is what makes a dot product a cosine similarity.
func (e *Embedder) Embed(texts []string) ([][]float32, error) {
	return e.embed(texts, true)
}

// EmbedRaw is Embed without normalization.
func (e *Embedder) EmbedRaw(texts []string) ([][]float32, error) {
	return e.embed(texts, false)
}

func (e *Embedder) embed(texts []string, normalize bool) ([][]float32, error) {
	if e.handle == 0 {
		return nil, &Error{Code: "ENGINE_CLOSED", Message: "this Embedder has already been closed"}
	}
	payload, err := json.Marshal(map[string]any{"input": texts, "normalize": normalize})
	if err != nil {
		return nil, fmt.Errorf("modelnexus: could not encode request: %w", err)
	}
	raw := takeString(llbEmbed(e.handle, string(payload)))

	var res embedResponse
	if err := json.Unmarshal([]byte(raw), &res); err != nil {
		return nil, fmt.Errorf("modelnexus: unparseable embedding response: %w", err)
	}
	if res.Type == "error" {
		if res.Error != nil {
			return nil, res.Error
		}
		return nil, &Error{Code: "UNKNOWN", Message: "core reported an error with no detail"}
	}
	return res.Embeddings, nil
}

// Rerank scores documents against a query, best first.
//
// topN <= 0 returns every document. Requires a reranker model opened with
// PoolingRank; otherwise this fails with POOLING_NOT_RANK rather than returning
// numbers that look like scores and are not.
func (e *Embedder) Rerank(query string, documents []string, topN int) ([]RerankHit, error) {
	if e.handle == 0 {
		return nil, &Error{Code: "ENGINE_CLOSED", Message: "this Embedder has already been closed"}
	}
	request := map[string]any{"query": query, "documents": documents}
	if topN > 0 {
		request["top_n"] = topN
	}
	payload, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("modelnexus: could not encode request: %w", err)
	}
	raw := takeString(llbRerank(e.handle, string(payload)))

	var res rerankResponse
	if err := json.Unmarshal([]byte(raw), &res); err != nil {
		return nil, fmt.Errorf("modelnexus: unparseable rerank response: %w", err)
	}
	if res.Type == "error" {
		if res.Error != nil {
			return nil, res.Error
		}
		return nil, &Error{Code: "UNKNOWN", Message: "core reported an error with no detail"}
	}
	return res.Results, nil
}

// Close releases the model and its context. Safe to call more than once.
func (e *Embedder) Close() error {
	if e.handle == 0 {
		return nil
	}
	llbEmbedDestroy(e.handle)
	e.handle = 0
	releaseSink(eventSink, e.eventID)
	return nil
}

// ---------------------------------------------------------------- logging

// LogLevel is how much the inference engine is allowed to say.
//
// The bridge defaults to LogWarn rather than llama.cpp's own default: a library
// embedded in someone else's process should be quiet unless asked.
type LogLevel int32

const (
	// LogNone silences the engine entirely.
	LogNone  LogLevel = 0
	LogDebug LogLevel = 1
	LogInfo  LogLevel = 2
	// LogWarn is the default.
	LogWarn  LogLevel = 3
	LogError LogLevel = 4
)

// SetLogLevel sets how much the engine logs.
//
// Call it before loading a model: llama.cpp starts logging during load, so
// afterwards is too late to silence it.
func SetLogLevel(level LogLevel) error {
	if err := ensureLoaded(); err != nil {
		return err
	}
	llbSetLogLevel(int32(level))
	return nil
}

// The core retains the log callback until it is replaced, and purego trampolines
// are never freed -- so there is exactly ONE, registered on first use, with the
// current handler swapped behind a mutex.
var (
	logMu          sync.Mutex
	logHandler     func(LogLevel, string)
	logTrampoline  uintptr
	logRegisterOne sync.Once
)

// SetLogHandler routes engine log output to fn instead of stderr.
// Passing nil restores stderr.
func SetLogHandler(fn func(level LogLevel, text string)) error {
	if err := ensureLoaded(); err != nil {
		return err
	}
	logMu.Lock()
	logHandler = fn
	logMu.Unlock()

	if fn == nil {
		llbSetLogCallback(0, 0)
		return nil
	}
	logRegisterOne.Do(func() {
		logTrampoline = purego.NewCallback(func(level int32, text unsafe.Pointer, _ uintptr) uintptr {
			logMu.Lock()
			h := logHandler
			logMu.Unlock()
			if h != nil {
				h(LogLevel(level), goString(text))
			}
			return 0
		})
	})
	llbSetLogCallback(logTrampoline, 0)
	return nil
}
