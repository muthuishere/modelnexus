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
}

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
	tokenSink = map[uintptr]func(string){}

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
		tokenTrampoline = purego.NewCallback(func(piece unsafe.Pointer, user uintptr) uintptr {
			cbMu.Lock()
			fn := tokenSink[user]
			cbMu.Unlock()
			if fn != nil {
				fn(goString(piece))
			}
			return 0
		})
	})
}

func registerSink(m map[uintptr]func(string), fn func(string)) uintptr {
	cbMu.Lock()
	defer cbMu.Unlock()
	cbNextID++
	id := cbNextID
	m[id] = fn
	return id
}

func releaseSink(m map[uintptr]func(string), id uintptr) {
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

type openConfig struct{ onEvent func(string) }

// WithEventHandler receives the core's progress events during load and inference.
func WithEventHandler(fn func(string)) Option {
	return func(c *openConfig) { c.onEvent = fn }
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

	handle := llbChatCreate(ggufPath, eventTrampoline, chat.eventID)
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
	return c.infer(req, nil)
}

// InferStream runs one turn, calling onToken with each decoded piece as it is
// produced. The complete response is still returned when generation finishes.
func (c *Chat) InferStream(req Request, onToken func(string)) (*Response, error) {
	return c.infer(req, onToken)
}

func (c *Chat) infer(req Request, onToken func(string)) (*Response, error) {
	if c.handle == 0 {
		return nil, &Error{Code: "ENGINE_CLOSED", Message: "this Chat has already been closed"}
	}
	payload, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("modelnexus: could not encode request: %w", err)
	}

	var raw string
	if onToken == nil {
		raw = takeString(llbChatInfer(c.handle, string(payload)))
	} else {
		id := registerSink(tokenSink, onToken)
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
