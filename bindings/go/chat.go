// Package modelnexus runs a local LLM inside your own process.
//
//	chat, err := modelnexus.Open("model.gguf")
//	defer chat.Close()
//
//	resp, err := chat.Infer(modelnexus.Request{
//	    Messages: []modelnexus.Message{{Role: "user", Content: "hello"}},
//	})
//	fmt.Println(resp.Text)
//
// No server, no subprocess, no port -- and no cgo: the native bridge is loaded at
// runtime with purego, so CGO_ENABLED=0 and cross-compilation both keep working.
package modelnexus

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Message is one turn of a conversation.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

// ToolCall is a call the model proposes. It is never executed here -- modelnexus
// emits OpenAI-shaped tool calls and stops. Executing them is the caller's job
// (ADR-0003).
type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool is a function the model may call, in OpenAI's schema shape.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction describes one callable.
type ToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// Request is one inference call.
//
// Generation parameters are pointers so that "unset" is distinguishable from the
// zero value -- a Temperature of 0 is a legitimate, and very different, request
// from "use the default".
type Request struct {
	Messages   []Message `json:"messages"`
	Tools      []Tool    `json:"tools,omitempty"`
	ToolChoice string    `json:"tool_choice,omitempty"`

	Temperature   *float64 `json:"temperature,omitempty"`
	TopK          *int     `json:"top_k,omitempty"`
	TopP          *float64 `json:"top_p,omitempty"`
	MinP          *float64 `json:"min_p,omitempty"`
	MaxTokens     *int     `json:"max_tokens,omitempty"`
	RepeatPenalty *float64 `json:"repeat_penalty,omitempty"`
	Seed          *uint32  `json:"seed,omitempty"`
	Stop          []string `json:"stop,omitempty"`

	// JSONSchema constrains the output to a JSON Schema. The returned Text is
	// guaranteed to parse as JSON: the grammar llama.cpp derives from a schema
	// deliberately permits a ```json fence, and the core strips it before returning.
	//
	// Any value that marshals to a JSON Schema object works -- a map, or a struct.
	JSONSchema any `json:"json_schema,omitempty"`

	// Grammar constrains the output to a raw GBNF grammar.
	//
	// Setting this together with JSONSchema is rejected by the core with
	// INVALID_REQUEST rather than resolved by precedence: a silent winner between
	// two output constraints is a debugging session nobody should have.
	Grammar string `json:"grammar,omitempty"`

	// ReuseCache controls KV prefix reuse across calls on the same Chat. The core's
	// default is true, which is why this is a pointer -- leaving it nil means "the
	// core decides", and a plain bool would silently opt every caller out.
	//
	// Reuse is purely a latency property; output is identical either way. Set it to
	// false when calls must be provably independent (a determinism harness, or
	// tenants sharing one handle).
	ReuseCache *bool `json:"reuse_cache,omitempty"`
}

// ReuseCacheOff is a convenience for Request.ReuseCache, which needs an address.
func ReuseCacheOff() *bool { off := false; return &off }

// ReuseCacheOn is a convenience for Request.ReuseCache. It states the core's default
// explicitly, which is worth doing in a test that is about the flag.
func ReuseCacheOn() *bool { on := true; return &on }

// Usage is the token accounting for one call.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Response is the result of one inference call.
type Response struct {
	Type         string     `json:"type"`
	Text         string     `json:"text"`
	ToolCalls    []ToolCall `json:"tool_calls"`
	FinishReason string     `json:"finish_reason"`
	Usage        Usage      `json:"usage"`
}

// FinishCancelled is the FinishReason of a generation a consumer stopped.
//
// A cancelled generation is a RESULT, not an error: the response carries the text
// produced so far and honest usage counts, because those tokens were really
// generated. Nothing here turns it into a Go error -- the caller decides.
const FinishCancelled = "cancelled"

// Cancelled reports whether generation was stopped early by the token callback or a
// cancelled context.
func (r *Response) Cancelled() bool { return r.FinishReason == FinishCancelled }

// TokenCount is what a message list will cost, without running inference.
type TokenCount struct {
	// Tokens is the prompt length after the model's chat template is applied.
	Tokens int `json:"tokens"`
	// NCtx is the engine's context window, so a caller can compare the two.
	NCtx int `json:"n_ctx"`
}

// ModelInfo reports a model's tool-calling capability.
type ModelInfo struct {
	SupportsTools      bool   `json:"supports_tools"`
	SupportsToolCalls  bool   `json:"supports_tool_calls"`
	HasToolUseTemplate bool   `json:"has_tool_use_template"`
	ChatFormat         string `json:"chat_format"`
	Error              string `json:"error"`
}

// Error is a failure reported by the core.
//
// Code is stable and identical across every language binding; Message is for humans.
type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *Error) Error() string {
	if e.Code == "" {
		return "modelnexus: " + e.Message
	}
	return "modelnexus: " + e.Code + ": " + e.Message
}

// ---------------------------------------------------------------- callbacks

// purego.NewCallback allocates a trampoline that is NEVER released, and the
// platform caps how many can exist (a few thousand). Creating one per Chat would
// therefore leak and eventually fail outright in a long-lived process.
//
// So there is exactly ONE trampoline per callback type, forever, and the
// per-instance state is reached through the user_data pointer -- which carries an
// integer key into this registry rather than a Go pointer, because native code must
// never hold a pointer the Go garbage collector can move.
var (
	cbMu      sync.Mutex
	cbNextID  uintptr
	eventSink = map[uintptr]func(string){}
	tokenSink = map[uintptr]func(string) bool{}

	eventTrampoline uintptr
	tokenTrampoline uintptr
	trampolineOnce  sync.Once
)

func initTrampolines() {
	trampolineOnce.Do(func() {
		eventTrampoline = purego.NewCallback(func(msg unsafe.Pointer, user uintptr) uintptr {
			cbMu.Lock()
			fn := eventSink[user]
			cbMu.Unlock()
			if fn != nil {
				fn(goString(msg))
			}
			return 0
		})
		// The ABI inverts Go's sense here: the C callback returns non-zero to STOP,
		// while a Go consumer says "keep going" with true. The inversion lives in
		// this one line so no consumer has to remember it.
		tokenTrampoline = purego.NewCallback(func(piece unsafe.Pointer, user uintptr) uintptr {
			cbMu.Lock()
			fn := tokenSink[user]
			cbMu.Unlock()
			if fn != nil && !fn(goString(piece)) {
				return 1
			}
			return 0
		})
	})
}

func registerSink[T any](m map[uintptr]T, fn T) uintptr {
	cbMu.Lock()
	defer cbMu.Unlock()
	cbNextID++
	id := cbNextID
	m[id] = fn
	return id
}

func releaseSink[T any](m map[uintptr]T, id uintptr) {
	cbMu.Lock()
	defer cbMu.Unlock()
	delete(m, id)
}

// ---------------------------------------------------------------- API

// Version reports the bridge version and the llama.cpp tag it was linked against.
func Version() (string, error) {
	if err := ensureLoaded(); err != nil {
		return "", err
	}
	return goString(llbVersion()), nil
}

// Info inspects a GGUF's tool-calling capability without loading an engine.
//
// Cheap enough to call before committing to a multi-gigabyte load, which is why the
// core exposes it separately from Open.
func Info(ggufPath string) (ModelInfo, error) {
	if err := ensureLoaded(); err != nil {
		return ModelInfo{}, err
	}
	raw := takeString(llbModelInfo(ggufPath))
	var info ModelInfo
	if err := json.Unmarshal([]byte(raw), &info); err != nil {
		return ModelInfo{}, fmt.Errorf("modelnexus: unparseable model info: %w", err)
	}
	if info.Error != "" {
		return info, &Error{Code: "MODEL_INFO_FAILED", Message: info.Error}
	}
	return info, nil
}

// Chat is a loaded model and its inference context. Close it when done.
type Chat struct {
	handle  uintptr
	eventID uintptr
	mu      sync.Mutex
	events  []string
}

// Option configures Open.
type Option func(*openConfig)

type openConfig struct {
	onEvent func(string)
	// Context parameters, marshalled to the core's create config. Zero means "not
	// set" and is omitted, so an Open with no options sends an empty config and the
	// core applies exactly the defaults it used before the parameter existed.
	nCtx, nBatch, nSeqMax int
}

// WithEventHandler receives the core's progress events during load and inference.
func WithEventHandler(fn func(string)) Option {
	return func(c *openConfig) { c.onEvent = fn }
}

// WithContextSize sets the engine's context window in tokens. Default 4096.
//
// This is create-time: it is the size of the KV cache, and it cannot be changed on a
// live engine.
func WithContextSize(n int) Option {
	return func(c *openConfig) { c.nCtx = n }
}

// WithBatchSize sets the logical batch size. Default 512.
func WithBatchSize(n int) Option {
	return func(c *openConfig) { c.nBatch = n }
}

// WithMaxSequences reserves room for n concurrent sequences. Default 1.
//
// It has no observable effect today. It is accepted now because it is a create-time
// parameter -- the one kind that cannot be added later through request JSON -- so
// reserving it is what lets multi-sequence slots arrive without an ABI break
// (ADR-0008 D6).
func WithMaxSequences(n int) Option {
	return func(c *openConfig) { c.nSeqMax = n }
}

// Open loads a GGUF model and creates an inference engine.
//
// Models whose chat template cannot do tool calling are rejected here rather than
// silently degraded -- a deliberate contract inherited from the core.
func Open(ggufPath string, opts ...Option) (*Chat, error) {
	if err := ensureLoaded(); err != nil {
		return nil, err
	}
	initTrampolines()

	var cfg openConfig
	for _, o := range opts {
		o(&cfg)
	}
	config := map[string]any{}
	if cfg.nCtx > 0 {
		config["n_ctx"] = cfg.nCtx
	}
	if cfg.nBatch > 0 {
		config["n_batch"] = cfg.nBatch
	}
	if cfg.nSeqMax > 0 {
		config["n_seq_max"] = cfg.nSeqMax
	}
	configJSON, err := json.Marshal(config)
	if err != nil {
		return nil, fmt.Errorf("modelnexus: could not encode config: %w", err)
	}

	chat := &Chat{}
	// The core STORES this callback on the engine and calls it for the whole life of
	// the handle, not just during create -- so the registration must outlive Open.
	// It is released in Close.
	chat.eventID = registerSink(eventSink, func(msg string) {
		chat.mu.Lock()
		chat.events = append(chat.events, msg)
		chat.mu.Unlock()
		if cfg.onEvent != nil {
			cfg.onEvent(msg)
		}
	})

	handle := llbChatCreate(ggufPath, string(configJSON), eventTrampoline, chat.eventID)
	if handle == 0 {
		defer releaseSink(eventSink, chat.eventID)
		// NULL is the one place the core signals failure with a null pointer; the
		// reason arrives through the event callback instead.
		chat.mu.Lock()
		events := append([]string(nil), chat.events...)
		chat.mu.Unlock()
		for _, e := range events {
			if e == "create_failure:tools_unsupported" {
				return nil, &Error{
					Code:    "MODEL_NOT_TOOL_CAPABLE",
					Message: ggufPath + " has no tool-calling chat template",
				}
			}
		}
		detail := "unknown reason"
		if len(events) > 0 {
			detail = fmt.Sprint(events)
		}
		return nil, &Error{Code: "MODEL_LOAD_FAILED", Message: "could not load " + ggufPath + " (" + detail + ")"}
	}
	chat.handle = handle
	return chat, nil
}

// Infer runs one turn and returns the response.
func (c *Chat) Infer(req Request) (*Response, error) {
	return c.infer(context.Background(), req, nil)
}

// InferStream runs one turn, calling onToken with each decoded piece as it is
// produced. The complete response is still returned when generation finishes.
//
// onToken returns whether to KEEP GOING: return false to stop generation, which ends
// the turn with FinishCancelled and a complete response carrying the text produced so
// far. Stopping is a result, not an error.
func (c *Chat) InferStream(req Request, onToken func(piece string) bool) (*Response, error) {
	return c.infer(context.Background(), req, onToken)
}

// InferContext is Infer, stopped by cancelling ctx.
//
// The ABI's only cancellation mechanism is the token callback's return value, so ctx
// is checked once per decoded token: cancellation takes effect before the NEXT token,
// not mid-token. A cancelled run returns a normal response with FinishCancelled and a
// nil error -- the tokens were really generated and the usage counts are real, and
// throwing that away to return ctx.Err() would lose the one thing the caller was
// billed for. Check Response.Cancelled, or ctx.Err(), to tell the two apart.
func (c *Chat) InferContext(ctx context.Context, req Request) (*Response, error) {
	return c.infer(ctx, req, nil)
}

// InferStreamContext is InferStream, additionally stopped by cancelling ctx.
// Either the callback returning false or ctx being cancelled ends generation.
func (c *Chat) InferStreamContext(ctx context.Context, req Request, onToken func(piece string) bool) (*Response, error) {
	return c.infer(ctx, req, onToken)
}

func (c *Chat) infer(ctx context.Context, req Request, onToken func(string) bool) (*Response, error) {
	if c.handle == 0 {
		return nil, &Error{Code: "ENGINE_CLOSED", Message: "this Chat has already been closed"}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	payload, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("modelnexus: could not encode request: %w", err)
	}

	// A context that can never be cancelled needs no per-token check, so it takes
	// the plain entry point -- the streaming one exists to carry the callback.
	cancellable := ctx.Done() != nil
	var raw string
	if onToken == nil && !cancellable {
		raw = takeString(llbChatInfer(c.handle, string(payload)))
	} else {
		// The piece is delivered BEFORE the context is checked, deliberately: it has
		// already been generated and is already counted in the response's usage, so
		// withholding it would leave a consumer's accumulated text one token short of
		// the Response.Text it is handed at the end.
		step := func(piece string) bool {
			keepGoing := true
			if onToken != nil {
				keepGoing = onToken(piece)
			}
			if cancellable && ctx.Err() != nil {
				return false
			}
			return keepGoing
		}
		id := registerSink(tokenSink, step)
		defer releaseSink(tokenSink, id)
		raw = takeString(llbChatStream(c.handle, string(payload), tokenTrampoline, id))
	}

	var probe struct {
		Type  string `json:"type"`
		Error *Error `json:"error"`
	}
	if err := json.Unmarshal([]byte(raw), &probe); err != nil {
		return nil, fmt.Errorf("modelnexus: unparseable response: %w", err)
	}
	if probe.Type == "error" {
		if probe.Error != nil {
			return nil, probe.Error
		}
		return nil, &Error{Code: "UNKNOWN", Message: "core reported an error with no detail"}
	}

	var resp Response
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		return nil, fmt.Errorf("modelnexus: unparseable response: %w", err)
	}
	return &resp, nil
}

// CountTokens reports what a request's messages will cost, without generating.
//
// It applies the model's chat template and tokenizes; it creates no context, decodes
// nothing, and does not disturb the KV cache, so it is safe to call between
// inferences. Every generation parameter on req is ignored.
//
// This is an ABI call rather than a Go helper because counting needs both the model's
// vocabulary and its parsed chat template, and a binding holds neither.
func (c *Chat) CountTokens(req Request) (TokenCount, error) {
	if c.handle == 0 {
		return TokenCount{}, &Error{Code: "ENGINE_CLOSED", Message: "this Chat has already been closed"}
	}
	payload, err := json.Marshal(req)
	if err != nil {
		return TokenCount{}, fmt.Errorf("modelnexus: could not encode request: %w", err)
	}
	raw := takeString(llbCountTokens(c.handle, string(payload)))

	var res struct {
		Type string `json:"type"`
		TokenCount
		Error *Error `json:"error"`
	}
	if err := json.Unmarshal([]byte(raw), &res); err != nil {
		return TokenCount{}, fmt.Errorf("modelnexus: unparseable token count: %w", err)
	}
	if res.Type == "error" {
		if res.Error != nil {
			return TokenCount{}, res.Error
		}
		return TokenCount{}, &Error{Code: "UNKNOWN", Message: "core reported an error with no detail"}
	}
	return res.TokenCount, nil
}

// Close releases the model and its context. Safe to call more than once.
func (c *Chat) Close() error {
	if c.handle == 0 {
		return nil
	}
	llbChatDestroy(c.handle)
	c.handle = 0
	releaseSink(eventSink, c.eventID)
	return nil
}
