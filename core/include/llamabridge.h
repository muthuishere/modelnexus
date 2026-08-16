/*
 * llamabridge.h
 *
 * Thin C ABI over llama.cpp for use from Java (Panama FFM).
 *
 * Design goals:
 *   - Pure C surface, no C++ name-mangling, easy to bind from any FFI.
 *     (The implementation is now C++ — it uses llama.cpp's common_chat /
 *     common_sampler APIs — but everything below is exported as extern "C".)
 *   - Opaque handle pattern so the Java side never sees llama.cpp types.
 *   - Strings cross the boundary as malloc'd UTF-8 — caller frees via
 *     llb_string_free. This avoids requiring callers to allocate
 *     output buffers up front and keeps ownership obvious.
 *   - Single-call inference: request JSON in, response JSON out.
 *   - Optional token streaming via a caller-supplied callback.
 *
 * Build artefact: libllamabridge.dylib (linked against libllama + llama-common).
 */

#ifndef LLAMABRIDGE_H
#define LLAMABRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* ------------------------------------------------------------------ */
/* Opaque chat handle                                                  */
/* ------------------------------------------------------------------ */

typedef struct llb_chat llb_chat_t;

/* ------------------------------------------------------------------ */
/* Event callback                                                      */
/*                                                                     */
/* The bridge invokes this callback with short, NUL-terminated event   */
/* strings during model load and inference. The pointer is valid only  */
/* for the duration of the callback. user_data is forwarded verbatim   */
/* from the create call. May be NULL.                                  */
/* ------------------------------------------------------------------ */

typedef void (*llb_event_cb)(const char* event, void* user_data);

/* ------------------------------------------------------------------ */
/* Token streaming callback                                            */
/*                                                                     */
/* Invoked once per decoded token piece (NUL-terminated UTF-8) during  */
/* llb_chat_infer_stream. The pointer is valid only for the duration   */
/* of the callback. user_data is forwarded verbatim. May be NULL.      */
/*                                                                     */
/* LIFETIME — and this differs from llb_event_cb / llb_log_cb, which the  */
/* core RETAINS. This callback is used ONLY for the duration of the      */
/* llb_chat_infer_stream call that was given it, and is not stored. A    */
/* binding may therefore hold its trampoline in a per-call local; it     */
/* does NOT need to keep it alive for the life of the handle.            */
/*                                                                       */
/* RETURN VALUE (breaking change in 0.2.0 — was void):                 */
/*   0        continue generating                                      */
/*   non-zero STOP. Generation ends before the next token is decoded.  */
/*                                                                     */
/* A callback MUST NOT let a host-language exception escape into the     */
/* core. The bridge is C++ with its own unwinding, and a managed or      */
/* interpreted exception crossing this frame is undefined behaviour.     */
/* Catch it, return non-zero to stop, and re-raise after the call        */
/* returns — the cancellation path is exactly the mechanism for this.    */
/*                                                                       */
/* Cancelling is a RESULT, not an error: the call still returns a       */
/* complete response carrying the text produced so far, honest usage    */
/* counts, and "finish_reason": "cancelled".                           */
/*                                                                     */
/* The KV cache is rolled back to the prompt boundary on cancellation,  */
/* so the abandoned partial turn can never be matched as a reusable     */
/* prefix by a later call (see llb_chat_create's "reuse_cache").        */
/* ------------------------------------------------------------------ */

typedef int (*llb_token_cb)(const char* token_piece, void* user_data);

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

/*
 * Load a GGUF chat model from disk and create an inference engine.
 *
 * mochallama enforces a tool-calling policy: a model whose chat template
 * does NOT support tool calling cannot produce a usable engine. After the
 * chat templates are built, caps are computed from the `tool_use` template
 * variant when the GGUF ships one (else the default template). If
 * `supports_tools` is false, the bridge frees everything, emits the event
 * "create_failure:tools_unsupported" via event_cb, and returns NULL.
 *
 * Configuration (breaking change in 0.2.0 — this parameter is new).
 * config_json is an optional JSON object; NULL or "" means defaults, which
 * are exactly the values this function used before the parameter existed.
 * Mirrors llb_embed_create's shape.
 *
 *   {
 *     "n_ctx":     N,   // context size            (default 4096)
 *     "n_batch":   N,   // logical batch size      (default 512)
 *     "n_seq_max": N    // max concurrent sequences (default 1)
 *   }
 *
 * n_seq_max has no observable effect today. It is accepted now because it is
 * a CREATE-TIME context parameter — the one thing that cannot be added later
 * through request JSON — and reserving it is what lets multi-sequence "slots"
 * be added without a future ABI break.
 *
 * @param gguf_path   Path to the .gguf model file.
 * @param config_json Optional JSON config (NULL => defaults).
 * @param event_cb    Optional progress callback (NULL to disable).
 * @param user_data   Opaque pointer passed back to event_cb.
 * @return            Engine handle, or NULL on any failure
 *                    (missing file, bad model, OOM, context init failure,
 *                    or a chat template that does not support tool calling —
 *                    distinguished by the "create_failure:tools_unsupported"
 *                    event).
 */
llb_chat_t* llb_chat_create(const char* gguf_path,
                            const char* config_json,
                            llb_event_cb event_cb,
                            void* user_data);

/*
 * Inspect a GGUF model's tool-calling capability WITHOUT creating an engine.
 *
 * Loads just the model (no inference context), builds its chat templates and
 * computes caps from the `tool_use` template variant when present (else the
 * default), then frees the model again. Intended as a pre-flight check so
 * callers can reject non-tool-capable models before committing to a load.
 *
 * Returns a malloc'd, NUL-terminated UTF-8 JSON string — release it via
 * llb_string_free. NEVER returns NULL; failures are reported in the JSON's
 * "error" field. Shape:
 *
 *   {
 *     "supports_tools":       bool,     // caps.supports_tools (from tool_use variant if present)
 *     "supports_tool_calls":  bool,     // caps.supports_tool_calls
 *     "has_tool_use_template":bool,     // GGUF has tokenizer.chat_template.tool_use
 *     "chat_format":          "CONTENT_ONLY|PEG_SIMPLE|PEG_NATIVE|PEG_GEMMA4|...",
 *     "error":                null | "<reason>"
 *   }
 *
 * On error, the boolean fields are false, "chat_format" is null and "error"
 * carries a human-readable reason (e.g. "model_not_found", "load_model",
 * "chat_template").
 */
const char* llb_model_info(const char* gguf_path);

/*
 * Run a single chat inference (non-streaming).
 *
 * Request JSON shape (all generation params are optional with defaults):
 *   {
 *     "messages":      [ {"role":"system|user|assistant|tool","content":"..."}, ... ],
 *     "tools":         [ {"type":"function","function":{"name":"...","description":"...",
 *                                                       "parameters":{...json-schema...}}}, ... ],
 *     "tool_choice":   "auto|none|required",   // default auto
 *     "temperature":   T,    // default 0.7
 *     "top_k":         K,     // default 40
 *     "top_p":         P,     // default 0.95
 *     "min_p":         M,     // default 0.05
 *     "max_tokens":    N,     // default 256
 *     "repeat_penalty":R,     // default 1.0 (disabled)
 *     "seed":          S,     // default random (LLAMA_DEFAULT_SEED)
 *     "stop":          [ "...", ... ], // optional stop strings
 *
 *     // --- constrained output (0.2.0). At most ONE of these two. -------
 *     "json_schema":   {...},  // JSON Schema; output is constrained to it
 *     "grammar":       "...",  // raw GBNF; output is constrained to it
 *
 *     // --- KV cache (0.2.0) --------------------------------------------
 *     "reuse_cache":   bool    // default TRUE
 *   }
 *
 * CONSTRAINED OUTPUT. Supplying both "json_schema" and "grammar" is an error
 * (INVALID_REQUEST), not a precedence rule. With "json_schema" the returned
 * content is guaranteed to PARSE as JSON and satisfy the schema: the grammar
 * llama.cpp generates deliberately permits a ```json markdown fence, so the
 * bridge strips it before returning rather than handing callers a "JSON"
 * string that json.loads() rejects.
 *
 * CACHE REUSE. By default the engine keeps the tokens resident in its KV
 * cache between calls and re-decodes only the part of a new prompt that
 * differs — the longest common prefix is reused. An appending conversation
 * therefore costs a flat amount per turn instead of growing linearly, which
 * over a session is the difference between linear and quadratic total work.
 * Output is unaffected; this is purely latency. Set "reuse_cache": false when
 * each call must be provably independent (determinism harnesses, tenants
 * sharing one handle).
 *
 * Response JSON shape (success):
 *   {
 *     "type":  "assistant_text" | "tool_call",
 *     "text":  "...",
 *     "tool_calls": [ {"id":"...","name":"...","arguments":"{...}"}, ... ],
 *     "finish_reason": "stop" | "length" | "tool_calls",
 *     "usage": {
 *       "prompt_tokens":     N,
 *       "completion_tokens": N,
 *       "total_tokens":      N
 *     }
 *   }
 *
 * Response JSON shape (failure):
 *   {
 *     "type":  "error",
 *     "error": { "code": "...", "message": "..." }
 *   }
 *
 * The returned pointer is heap-allocated. The caller MUST release it
 * via llb_string_free. Never returns NULL — errors are reported as
 * error JSON.
 */
const char* llb_chat_infer(llb_chat_t* chat, const char* request_json);

/*
 * Run a single chat inference, streaming each decoded token piece to
 * token_cb as it is produced. Accepts the same request JSON as
 * llb_chat_infer and returns the SAME final response JSON (with usage +
 * tool_calls) once generation completes. token_cb may be NULL (then this
 * behaves like llb_chat_infer). The returned pointer is heap-allocated
 * and must be released via llb_string_free.
 */
const char* llb_chat_infer_stream(llb_chat_t* chat, const char* request_json,
                                  llb_token_cb token_cb, void* user_data);

/*
 * Release a string previously returned by llb_chat_infer /
 * llb_chat_infer_stream. Safe to call on NULL.
 */
void llb_string_free(const char* s);

/*
 * Count the tokens a message list will occupy, WITHOUT running inference.
 *
 * Applies the model's chat template and tokenizes the result. Creates no
 * context, decodes nothing, and does not touch the KV cache — calling this
 * between two inferences cannot disturb cache reuse.
 *
 * Accepts the same "messages" (and optional "tools") shape as llb_chat_infer;
 * every generation parameter is ignored if present.
 *
 * Returns a malloc'd JSON string — release it via llb_string_free. Never
 * returns NULL; failures are error JSON.
 *
 *   { "type": "token_count", "tokens": N, "n_ctx": M }
 *
 * This lives in the ABI rather than in each binding because counting needs
 * BOTH the model's vocabulary and its parsed chat template, and a binding
 * holds neither.
 */
const char* llb_count_tokens(llb_chat_t* chat, const char* request_json);

/*
 * Inspect or drop the engine's KV cache.
 *
 * ONE entry point with a JSON op, like llb_chat_lora — so a new operation
 * costs no symbol and no binding change.
 *
 *   {"op":"status"}   report what is resident. Changes nothing.
 *   {"op":"clear"}    drop it. Frees the KV memory and forgets the sequence.
 *
 * Both return:
 *
 *   { "type": "cache", "tokens": N, "n_ctx": M }
 *
 * where "tokens" is what is resident AFTER the operation — so a clear always
 * reports 0, and a caller can assert on the result rather than trust it.
 *
 * WHY THIS EXISTS. Reuse is on by default and keyed on the longest common
 * prefix, which is exactly right for a conversation that appends. It is
 * exactly wrong when the handle moves to UNRELATED work: the old conversation
 * keeps occupying context memory until something happens to evict it, and two
 * tenants sharing a handle would share a cache. Passing "reuse_cache": false
 * on the next inference also clears, but only as a side effect of doing work —
 * that is no help when the point is to release memory NOW, or when you want to
 * assert the cache is empty before handing the handle on.
 *
 * Returns a malloc'd JSON string — release it via llb_string_free. Never
 * returns NULL; failures are error JSON.
 */
const char* llb_chat_cache(llb_chat_t* chat, const char* request_json);

/*
 * Tear down an engine and release its model + context.
 * Safe to call on NULL.
 */
void llb_chat_destroy(llb_chat_t* chat);

/* ------------------------------------------------------------------ */
/* Logging                                                             */
/*                                                                     */
/* llama.cpp writes to stderr by default -- hundreds of lines per model */
/* load. That is fine for a CLI and unacceptable for a library embedded */
/* in someone else's application, which is what this is. So the bridge  */
/* installs its own log sink and gives the host control over it.        */
/*                                                                     */
/* Levels match ggml's: 0 none, 1 debug, 2 info, 3 warn, 4 error.       */
/* The bridge DEFAULTS TO WARN, not the engine's default -- a library   */
/* should be quiet unless asked, and a host that wants the firehose can */
/* ask for it.                                                          */
/*                                                                     */
/* Call llb_set_log_level(LLB_LOG_NONE) to silence the engine entirely. */
/* ------------------------------------------------------------------ */

#define LLB_LOG_NONE  0
#define LLB_LOG_DEBUG 1
#define LLB_LOG_INFO  2
#define LLB_LOG_WARN  3
#define LLB_LOG_ERROR 4

/*
 * Set the minimum level the engine emits. Messages below it are dropped.
 * Safe to call at any time, including before any model is loaded.
 */
void llb_set_log_level(int level);

/*
 * Route engine log output to a callback instead of stderr.
 *
 * The callback receives the level and a NUL-terminated message; the pointer
 * is valid only for the duration of the call. Pass NULL to restore stderr.
 * The level filter set by llb_set_log_level still applies.
 *
 * The bridge RETAINS this pointer until it is replaced or cleared, so the
 * caller must keep the callable alive for that whole period -- the same rule
 * as llb_chat_create's event callback.
 */
typedef void (*llb_log_cb)(int level, const char* text, void* user_data);

void llb_set_log_callback(llb_log_cb cb, void* user_data);

/* ------------------------------------------------------------------ */
/* LoRA adapters                                                       */
/*                                                                     */
/* One entry point for every adapter operation, dispatched on an "op"  */
/* field in the request JSON. Adding an operation therefore does not   */
/* change the ABI and does not churn five language bindings -- the     */
/* same reason generation parameters travel inside the request rather  */
/* than as C arguments.                                                */
/*                                                                     */
/* Adapters are loaded against the engine's model and applied to its   */
/* context. Scales are per-adapter and may be changed at any time      */
/* between inferences; several adapters can be active at once.         */
/*                                                                     */
/* Request shapes:                                                     */
/*   {"op":"load",   "path":"adapter.gguf", "scale":1.0}               */
/*   {"op":"set",    "id":0, "scale":0.5}                              */
/*   {"op":"remove", "id":0}                                           */
/*   {"op":"clear"}                                                    */
/*   {"op":"list"}                                                     */
/*                                                                     */
/* Response (success) -- always the full adapter set after the op, so  */
/* a caller never has to track state the core already knows:           */
/*   {                                                                 */
/*     "type": "lora",                                                 */
/*     "id":   0,            // the affected adapter, for "load" only  */
/*     "adapters": [ {"id":0,"path":"...","scale":1.0}, ... ]          */
/*   }                                                                 */
/*                                                                     */
/* Failures come back as the same error JSON every other call uses.    */
/* Never returns NULL. Release via llb_string_free.                    */
/* ------------------------------------------------------------------ */

const char* llb_chat_lora(llb_chat_t* chat, const char* request_json);

/* ------------------------------------------------------------------ */
/* Embeddings and reranking                                            */
/*                                                                     */
/* A SEPARATE handle from llb_chat, deliberately: embedding needs a    */
/* context created with embeddings enabled and a pooling type fixed at */
/* creation, and reranking needs LLAMA_POOLING_TYPE_RANK specifically. */
/* Neither can be toggled on a generation context after the fact, so   */
/* sharing one handle would mean silently reloading the model.         */
/* ------------------------------------------------------------------ */

typedef struct llb_embed llb_embed_t;

/*
 * Load a model for embedding or reranking.
 *
 * config_json (may be NULL or "" for defaults):
 *   {
 *     "pooling": "mean" | "cls" | "last" | "rank" | "none",
 *     "n_ctx":   512,      // 0 = model default
 *     "n_batch": 512
 *   }
 *
 * Use "rank" for a reranker model -- llb_rerank REQUIRES it, because the
 * classification head is only attached to the graph under that pooling
 * type. Embedding models want "mean" or "cls"; when omitted the model's
 * own default is used.
 *
 * Unlike llb_chat_create there is NO tool-calling gate here: an
 * embedding model has no chat template and rejecting it would be absurd.
 *
 * Returns NULL on failure, with the reason delivered through event_cb
 * ("create_failure:model_not_found", ":load_model", ":init_context").
 */
llb_embed_t* llb_embed_create(const char* gguf_path,
                              const char* config_json,
                              llb_event_cb event_cb,
                              void* user_data);

/*
 * Embed one or more texts.
 *
 * Request:
 *   {
 *     "input":     ["text one", "text two"],   // or a single string
 *     "normalize": true                        // default true (L2)
 *   }
 *
 * Response:
 *   {
 *     "type":       "embedding",
 *     "dim":        384,
 *     "embeddings": [ [ ... ], [ ... ] ],       // one per input, input order
 *     "usage":      { "prompt_tokens": N, "total_tokens": N }
 *   }
 *
 * Never returns NULL; failures are error JSON. Release via llb_string_free.
 */
const char* llb_embed(llb_embed_t* embed, const char* request_json);

/*
 * Rerank documents against a query. Requires a reranker model loaded with
 * "pooling":"rank" -- otherwise this returns error code POOLING_NOT_RANK
 * rather than silently returning meaningless scores.
 *
 * Request:
 *   {
 *     "query":     "what is the capital of France?",
 *     "documents": ["Paris is ...", "Berlin is ..."],
 *     "top_n":     2          // optional; default all, sorted by score desc
 *   }
 *
 * Response -- results are sorted by score descending and carry the ORIGINAL
 * index, so a caller can map back to its own list after reordering:
 *   {
 *     "type": "rerank",
 *     "results": [ {"index":0, "score":8.1}, {"index":1, "score":-4.2} ],
 *     "usage": { "prompt_tokens": N, "total_tokens": N }
 *   }
 *
 * Scores are raw model logits: comparable within one call, not calibrated
 * across models, and not probabilities.
 *
 * Never returns NULL. Release via llb_string_free.
 */
const char* llb_rerank(llb_embed_t* embed, const char* request_json);

/*
 * Tear down an embedding engine and release its model + context.
 * Safe to call on NULL.
 */
void llb_embed_destroy(llb_embed_t* embed);

/*
 * Return a static, NUL-terminated version string identifying the
 * bridge build (bridge version + linked llama.cpp tag). The returned
 * pointer is owned by the bridge — do NOT free it.
 */
const char* llb_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LLAMABRIDGE_H */
