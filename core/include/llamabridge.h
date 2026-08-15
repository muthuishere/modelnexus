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
/* ------------------------------------------------------------------ */

typedef void (*llb_token_cb)(const char* token_piece, void* user_data);

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
 * @param gguf_path   Path to the .gguf model file.
 * @param event_cb    Optional progress callback (NULL to disable).
 * @param user_data   Opaque pointer passed back to event_cb.
 * @return            Engine handle, or NULL on any failure
 *                    (missing file, bad model, OOM, context init failure,
 *                    or a chat template that does not support tool calling —
 *                    distinguished by the "create_failure:tools_unsupported"
 *                    event).
 */
llb_chat_t* llb_chat_create(const char* gguf_path,
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
 *     "stop":          [ "...", ... ]   // optional stop strings
 *   }
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
 * Tear down an engine and release its model + context.
 * Safe to call on NULL.
 */
void llb_chat_destroy(llb_chat_t* chat);

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
