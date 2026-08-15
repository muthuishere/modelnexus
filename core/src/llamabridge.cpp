/*
 * llamabridge.cpp
 *
 * C++ implementation of the C ABI declared in llamabridge.h.
 *
 * Unlike the original plain-C bridge (which used llama_chat_apply_template
 * plus a hand-rolled sampler), this version drives llama.cpp's higher-level
 * "common" layer:
 *
 *   - common_chat_templates_init / common_chat_templates_apply for prompt
 *     building, tool-call grammar synthesis and per-format parsing metadata.
 *   - common_sampler for the full sampling chain (penalties, top-k/p, min-p,
 *     temperature, grammar) honouring the request's generation parameters.
 *   - common_chat_parse for extracting assistant text AND tool calls from the
 *     generated text.
 *
 * The whole surface is still exported as extern "C" so the Java/Panama side
 * sees a clean, unmangled ABI. nlohmann/json (vendored with llama.cpp) is used
 * for request parsing and response building.
 *
 * All bridge-owned strings are allocated with malloc()/strdup and must be
 * released by the caller via llb_string_free.
 */

#include "../include/llamabridge.h"

#include "llama.h"

#include "common.h"
#include "chat.h"
#include "sampling.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

/* ------------------------------------------------------------------ */
/* Build-time identity                                                 */
/* ------------------------------------------------------------------ */

#ifndef LLB_BRIDGE_VERSION
#define LLB_BRIDGE_VERSION "0.2.0"
#endif

#ifndef LLB_LLAMA_TAG
#define LLB_LLAMA_TAG "unknown"
#endif

static const char LLB_VERSION_STR[] =
    "llamabridge " LLB_BRIDGE_VERSION " (llama.cpp " LLB_LLAMA_TAG ")";

/* ------------------------------------------------------------------ */
/* Opaque handle definition                                            */
/* ------------------------------------------------------------------ */

/* One loaded LoRA adapter and the scale it is currently applied at. */
struct llb_lora_slot {
    int                        id      = -1;
    std::string                path;
    float                      scale   = 1.0f;
    struct llama_adapter_lora* adapter = nullptr;
};

struct llb_chat {
    std::string                model_path;
    llb_event_cb               event_cb  = nullptr;
    void*                      user_data = nullptr;
    bool                       closed    = false;
    struct llama_model*        model     = nullptr;
    struct llama_context*      ctx       = nullptr;
    common_chat_templates_ptr  templates;   // owns the parsed chat template(s)
    std::vector<llb_lora_slot> loras;       // active adapters, in application order
    int                        next_lora_id = 0;
};

/* Embedding / reranking engine.

   Separate from llb_chat because embeddings require a context built with
   embeddings enabled and a pooling type chosen up front; reranking further
   requires LLAMA_POOLING_TYPE_RANK, which is what attaches the model's
   classification head to the graph. None of that can be switched on a
   generation context after creation. */
struct llb_embed {
    std::string             model_path;
    llb_event_cb            event_cb  = nullptr;
    void*                   user_data = nullptr;
    struct llama_model*     model     = nullptr;
    struct llama_context*   ctx       = nullptr;
    enum llama_pooling_type pooling   = LLAMA_POOLING_TYPE_UNSPECIFIED;
    int                     n_batch   = 512;
    int                     n_seq_max = 8;   /* sequences packed into one decode */
};

/* ------------------------------------------------------------------ */
/* Event helper                                                        */
/* ------------------------------------------------------------------ */

static void emit(const struct llb_chat* chat, const char* msg) {
    if (chat && chat->event_cb && msg) {
        chat->event_cb(msg, chat->user_data);
    }
}

static void emit_raw(llb_event_cb cb, void* user_data, const char* msg) {
    if (cb && msg) cb(msg, user_data);
}

/* ================================================================== */
/* Logging                                                             */
/* ================================================================== */

/* A library embedded in someone else's process must not print hundreds of lines
   to their stderr on every model load. The bridge therefore owns the engine's log
   sink from first use, defaults to WARN rather than the engine's own default, and
   lets the host redirect or silence it. */

static int         g_log_level    = LLB_LOG_WARN;
static llb_log_cb  g_log_cb       = nullptr;
static void*       g_log_user     = nullptr;
static bool        g_log_installed = false;

static void bridge_log_sink(enum ggml_log_level level, const char* text, void* /*user*/) {
    if (!text) return;
    const int lvl = (int)level;
    /* GGML_LOG_LEVEL_CONT continues the previous line, so it inherits that line's
       level rather than being filtered on its own -- filtering it independently
       would slice messages in half. */
    if (lvl != (int)GGML_LOG_LEVEL_CONT && lvl < g_log_level) return;
    if (g_log_cb) {
        g_log_cb(lvl, text, g_log_user);
    } else {
        fputs(text, stderr);
    }
}

/* Installed lazily from every entry point that can start the engine, because
   llama.cpp begins logging during model load and there is no earlier hook. */
static void ensure_log_sink() {
    if (g_log_installed) return;
    llama_log_set(bridge_log_sink, nullptr);
    g_log_installed = true;
}

extern "C" void llb_set_log_level(int level) {
    if (level < LLB_LOG_NONE)  level = LLB_LOG_NONE;
    if (level > LLB_LOG_ERROR) level = LLB_LOG_ERROR;
    /* NONE means "emit nothing", so the threshold must sit above every real level. */
    g_log_level = (level == LLB_LOG_NONE) ? (LLB_LOG_ERROR + 1) : level;
    ensure_log_sink();
}

extern "C" void llb_set_log_callback(llb_log_cb cb, void* user_data) {
    g_log_cb   = cb;
    g_log_user = user_data;
    ensure_log_sink();
}

/* ------------------------------------------------------------------ */
/* Response helpers — return malloc'd C strings owned by the caller     */
/* ------------------------------------------------------------------ */

static const char* dup_cstr(const std::string& s) {
    char* p = (char*)malloc(s.size() + 1);
    if (!p) return nullptr;
    memcpy(p, s.data(), s.size());
    p[s.size()] = '\0';
    return p;
}

static const char* build_error(const char* code, const std::string& message) {
    json j;
    j["type"] = "error";
    j["error"] = { {"code", code}, {"message", message} };
    return dup_cstr(j.dump());
}

/* ------------------------------------------------------------------ */
/* Request parsing                                                     */
/* ------------------------------------------------------------------ */

struct gen_params {
    float    temperature   = 0.7f;
    int32_t  top_k         = 40;
    float    top_p         = 0.95f;
    float    min_p         = 0.05f;
    int32_t  max_tokens    = 256;
    float    repeat_penalty = 1.0f;   // 1.0 = disabled
    uint32_t seed          = LLAMA_DEFAULT_SEED;
    std::vector<std::string> stop;
};

static common_chat_tool_choice parse_tool_choice(const std::string& s) {
    if (s == "none")     return COMMON_CHAT_TOOL_CHOICE_NONE;
    if (s == "required") return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    return COMMON_CHAT_TOOL_CHOICE_AUTO;
}

/* ------------------------------------------------------------------ */
/* Tool-capability helpers                                             */
/*                                                                     */
/* At llama.cpp b9371 the authoritative tool-capability signal is      */
/* jinja::caps, surfaced via common_chat_templates_get_caps (reads the */
/* DEFAULT template's caps). A model whose tool support lives in a      */
/* separate `tokenizer.chat_template.tool_use` variant would otherwise  */
/* report supports_tools=false on its (tool-less) default template, so  */
/* when the GGUF ships a tool_use variant we re-build a templates object */
/* whose default IS that variant and read caps from it. See            */
/* docs/specs/tool-calling-support.md §A.2 / §C.2.                      */
/* ------------------------------------------------------------------ */

struct tool_caps {
    bool supports_tools        = false;
    bool supports_tool_calls   = false;
    bool has_tool_use_template = false;
};

// Read supports_tools / supports_tool_calls from an already-built templates
// object via the public caps map.
static void read_caps_into(const common_chat_templates* tmpls, tool_caps& out) {
    if (!tmpls) return;
    std::map<std::string, bool> caps = common_chat_templates_get_caps(tmpls);
    auto it_tools = caps.find("supports_tools");
    auto it_calls = caps.find("supports_tool_calls");
    if (it_tools != caps.end()) out.supports_tools      = it_tools->second;
    if (it_calls != caps.end()) out.supports_tool_calls = it_calls->second;
}

// Build caps for a loaded model, preferring the `tool_use` template variant
// when the GGUF defines one (else the default template). `default_tmpls` is
// the already-built default templates object (from common_chat_templates_init
// with an empty override) and may be reused so we don't rebuild it.
static tool_caps compute_tool_caps(struct llama_model* model,
                                   const common_chat_templates* default_tmpls) {
    tool_caps tc;
    if (!model) return tc;

    const char* tool_use_src = llama_model_chat_template(model, "tool_use");
    tc.has_tool_use_template = (tool_use_src != nullptr && tool_use_src[0] != '\0');

    if (tc.has_tool_use_template) {
        // Build a templates object whose DEFAULT is the tool_use source, so the
        // public caps query reflects the variant's real capability.
        try {
            common_chat_templates_ptr variant =
                common_chat_templates_init(model, std::string(tool_use_src));
            if (variant) {
                read_caps_into(variant.get(), tc);
                return tc;
            }
        } catch (const std::exception&) {
            // fall through to the default template's caps
        }
    }

    read_caps_into(default_tmpls, tc);
    return tc;
}

// Map common_chat_format to the enum-style names used by llb_model_info's JSON
// ("CONTENT_ONLY", "PEG_SIMPLE", ...). Diagnostic only — NOT the capability
// gate (the format reflects which parser family fired for a given call, not
// whether the model supports tools). See spec §A.4.
static const char* chat_format_enum_name(common_chat_format fmt) {
    switch (fmt) {
        case COMMON_CHAT_FORMAT_CONTENT_ONLY: return "CONTENT_ONLY";
        case COMMON_CHAT_FORMAT_PEG_SIMPLE:   return "PEG_SIMPLE";
        case COMMON_CHAT_FORMAT_PEG_NATIVE:   return "PEG_NATIVE";
        case COMMON_CHAT_FORMAT_PEG_GEMMA4:   return "PEG_GEMMA4";
        default:                              return "UNKNOWN";
    }
}

// Probe the chat format by applying the template with a single sample tool.
// Diagnostic only; returns "UNKNOWN" on any failure.
static std::string probe_chat_format(const common_chat_templates* tmpls) {
    if (!tmpls) return "UNKNOWN";
    try {
        common_chat_templates_inputs inputs;
        common_chat_msg user;
        user.role    = "user";
        user.content = "ping";
        inputs.messages.push_back(user);
        common_chat_tool tool;
        tool.name        = "tool1";
        tool.description  = "a sample tool";
        tool.parameters  = "{\"type\":\"object\",\"properties\":{}}";
        inputs.tools.push_back(tool);
        inputs.tool_choice          = COMMON_CHAT_TOOL_CHOICE_AUTO;
        inputs.add_generation_prompt = true;
        inputs.use_jinja             = true;
        common_chat_params p = common_chat_templates_apply(
            const_cast<common_chat_templates*>(tmpls), inputs);
        return chat_format_enum_name(p.format);
    } catch (const std::exception&) {
        return "UNKNOWN";
    }
}

/* ------------------------------------------------------------------ */
/* Generation loop                                                     */
/*                                                                     */
/* Tokenizes the formatted prompt, decodes it, then samples up to       */
/* max_tokens. Streams each piece to token_cb (if non-NULL). Stops on    */
/* EOG, max_tokens, or any user stop string. Reports real prompt /       */
/* completion token counts.                                             */
/* ------------------------------------------------------------------ */

static bool run_generation(struct llb_chat*           chat,
                           const common_chat_params&  cparams,
                           const gen_params&          gp,
                           llb_token_cb               token_cb,
                           void*                      user_data,
                           std::string&               out_text,
                           int&                       prompt_tokens_out,
                           int&                       completion_tokens_out,
                           bool&                       hit_eog) {
    const llama_vocab* vocab = llama_model_get_vocab(chat->model);
    if (!vocab) return false;

    hit_eog = false;

    // Tokenize the formatted prompt (special tokens parsed, BOS per template).
    std::vector<llama_token> tokens =
        common_tokenize(vocab, cparams.prompt, /*add_special=*/true, /*parse_special=*/true);
    if (tokens.empty()) return false;

    prompt_tokens_out = (int)tokens.size();
    const int32_t n_tok = (int32_t)tokens.size();

    // Clear KV cache so each call is independent.
    llama_memory_t mem = llama_get_memory(chat->ctx);
    if (mem) llama_memory_clear(mem, true);

    int n_batch = (int)llama_n_batch(chat->ctx);
    if (n_batch <= 0) n_batch = 512;

    const bool has_enc = llama_model_has_encoder(chat->model);
    if (has_enc) {
        for (int32_t i = 0; i < n_tok; i += n_batch) {
            int32_t chunk = n_tok - i;
            if (chunk > n_batch) chunk = n_batch;
            llama_batch batch = llama_batch_get_one(tokens.data() + i, chunk);
            if (llama_encode(chat->ctx, batch) != 0) return false;
        }
        llama_token dec_start = llama_model_decoder_start_token(chat->model);
        if (dec_start == LLAMA_TOKEN_NULL) {
            dec_start = llama_vocab_bos(vocab);
        }
        llama_batch dec_batch = llama_batch_get_one(&dec_start, 1);
        if (llama_decode(chat->ctx, dec_batch) != 0) return false;
    } else {
        for (int32_t i = 0; i < n_tok; i += n_batch) {
            int32_t chunk = n_tok - i;
            if (chunk > n_batch) chunk = n_batch;
            llama_batch batch = llama_batch_get_one(tokens.data() + i, chunk);
            if (llama_decode(chat->ctx, batch) != 0) return false;
        }
    }

    // Build the common_sampler honouring the request's full parameter set,
    // plus any tool-call grammar produced by common_chat_templates_apply.
    common_params_sampling sparams;
    sparams.seed           = gp.seed;
    sparams.temp           = gp.temperature;
    sparams.top_k          = gp.top_k;
    sparams.top_p          = gp.top_p;
    sparams.min_p          = gp.min_p;
    sparams.penalty_repeat = gp.repeat_penalty;

    // Wire the chat-template grammar (tool calls / output format) through.
    if (!cparams.grammar.empty()) {
        sparams.grammar = { COMMON_GRAMMAR_TYPE_TOOL_CALLS, cparams.grammar };
    }
    sparams.grammar_lazy       = cparams.grammar_lazy;
    sparams.grammar_triggers   = cparams.grammar_triggers;
    sparams.generation_prompt  = cparams.generation_prompt;
    for (const auto& t : cparams.preserved_tokens) {
        auto ids = common_tokenize(vocab, t, /*add_special=*/false, /*parse_special=*/true);
        if (ids.size() == 1) sparams.preserved_tokens.insert(ids[0]);
    }

    common_sampler* smpl = common_sampler_init(chat->model, sparams);
    if (!smpl) return false;

    // Stop strings = user stops + any additional stops from the template.
    std::vector<std::string> stops = gp.stop;
    for (const auto& s : cparams.additional_stops) stops.push_back(s);

    int n_decoded = 0;
    while (n_decoded < gp.max_tokens) {
        llama_token tok = common_sampler_sample(smpl, chat->ctx, -1);
        common_sampler_accept(smpl, tok, /*is_generated=*/true);

        if (llama_vocab_is_eog(vocab, tok)) {
            hit_eog = true;
            break;
        }

        std::string piece = common_token_to_piece(vocab, tok, /*special=*/false);
        if (!piece.empty()) {
            out_text += piece;
            if (token_cb) token_cb(piece.c_str(), user_data);
        }

        n_decoded++;

        // Honour stop strings: truncate and finish if one appears.
        bool stopped = false;
        for (const auto& s : stops) {
            if (!s.empty()) {
                size_t pos = out_text.find(s);
                if (pos != std::string::npos) {
                    out_text.erase(pos);
                    stopped = true;
                    break;
                }
            }
        }
        if (stopped) break;

        llama_batch nb = llama_batch_get_one(&tok, 1);
        if (llama_decode(chat->ctx, nb) != 0) break;
    }

    common_sampler_free(smpl);
    completion_tokens_out = n_decoded;
    return true;
}

/* ------------------------------------------------------------------ */
/* Core inference (shared by streaming + non-streaming entry points)    */
/* ------------------------------------------------------------------ */

static const char* infer_impl(llb_chat_t*   chat,
                              const char*    request_json,
                              llb_token_cb   token_cb,
                              void*          user_data) {
    if (!chat || chat->closed) {
        return build_error("ENGINE_CLOSED", "chat engine is closed or NULL");
    }
    if (!request_json) {
        return build_error("INVALID_REQUEST", "request_json is NULL");
    }

    emit(chat, "infer_start");

    json req;
    try {
        req = json::parse(request_json);
    } catch (const std::exception& e) {
        return build_error("INVALID_REQUEST", std::string("malformed request JSON: ") + e.what());
    }

    if (!req.contains("messages") || !req["messages"].is_array() || req["messages"].empty()) {
        return build_error("INVALID_REQUEST", "messages array missing or empty");
    }

    // Generation parameters (all optional).
    gen_params gp;
    try {
        gp.temperature    = req.value("temperature", 0.7);
        gp.top_k          = req.value("top_k", 40);
        gp.top_p          = req.value("top_p", 0.95);
        gp.min_p          = req.value("min_p", 0.05);
        gp.max_tokens     = req.value("max_tokens", 256);
        gp.repeat_penalty = req.value("repeat_penalty", 1.0);
        if (req.contains("seed") && req["seed"].is_number()) {
            gp.seed = (uint32_t)(int64_t)req["seed"].get<int64_t>();
        }
        if (req.contains("stop")) {
            if (req["stop"].is_array()) {
                for (const auto& s : req["stop"]) {
                    if (s.is_string()) gp.stop.push_back(s.get<std::string>());
                }
            } else if (req["stop"].is_string()) {
                gp.stop.push_back(req["stop"].get<std::string>());
            }
        }
    } catch (const std::exception& e) {
        return build_error("INVALID_REQUEST", std::string("bad generation parameter: ") + e.what());
    }
    if (gp.max_tokens <= 0) gp.max_tokens = 256;
    if (gp.temperature < 0.0f) gp.temperature = 0.0f;

    // Build template inputs: messages + tools + tool_choice.
    common_chat_templates_inputs inputs;
    try {
        inputs.messages = common_chat_msgs_parse_oaicompat(req["messages"]);
        if (req.contains("tools") && !req["tools"].is_null()) {
            inputs.tools = common_chat_tools_parse_oaicompat(req["tools"]);
        }
        if (req.contains("tool_choice") && req["tool_choice"].is_string()) {
            inputs.tool_choice = parse_tool_choice(req["tool_choice"].get<std::string>());
        }
    } catch (const std::exception& e) {
        return build_error("INVALID_REQUEST", std::string("failed to parse messages/tools: ") + e.what());
    }
    inputs.add_generation_prompt = true;
    inputs.use_jinja             = true;

    // Apply chat template -> formatted prompt + grammar + parse format.
    common_chat_params cparams;
    try {
        cparams = common_chat_templates_apply(chat->templates.get(), inputs);
    } catch (const std::exception& e) {
        return build_error("INTERNAL_BRIDGE_ERROR", std::string("template apply failed: ") + e.what());
    }

    // Run generation.
    std::string out_text;
    int pt = 0, ct = 0;
    bool hit_eog = false;
    bool ok;
    try {
        ok = run_generation(chat, cparams, gp, token_cb, user_data,
                            out_text, pt, ct, hit_eog);
    } catch (const std::exception& e) {
        emit(chat, "infer_failure");
        return build_error("INFERENCE_FAILED", std::string("generation aborted: ") + e.what());
    }
    if (!ok) {
        emit(chat, "infer_failure");
        return build_error("INFERENCE_FAILED", "generation aborted");
    }

    // Parse the generated text into assistant content + tool calls.
    // The common_chat_parser_params(common_chat_params) constructor copies only
    // the format + generation_prompt; for the PEG-based formats (PEG_SIMPLE /
    // PEG_NATIVE / PEG_GEMMA4) we must also load the serialized PEG parser that
    // common_chat_templates_apply produced into cparams.parser — otherwise the
    // parser can't extract tool calls and leaves the raw markup in content.
    common_chat_msg parsed;
    try {
        common_chat_parser_params pp(cparams);
        pp.parse_tool_calls = true;
        if (!cparams.parser.empty()) {
            pp.parser.load(cparams.parser);
        }
        parsed = common_chat_parse(out_text, /*is_partial=*/false, pp);
    } catch (const std::exception&) {
        // Fall back to treating the raw text as plain content.
        parsed = common_chat_msg{};
        parsed.role    = "assistant";
        parsed.content = out_text;
    }

    // Build response JSON.
    json resp;
    json tool_calls = json::array();
    for (const auto& tc : parsed.tool_calls) {
        tool_calls.push_back({
            {"id", tc.id},
            {"name", tc.name},
            {"arguments", tc.arguments},
        });
    }

    std::string finish_reason;
    if (!parsed.tool_calls.empty()) {
        resp["type"]  = "tool_call";
        finish_reason = "tool_calls";
    } else {
        resp["type"]  = "assistant_text";
        finish_reason = hit_eog ? "stop" : (ct >= gp.max_tokens ? "length" : "stop");
    }

    resp["text"]          = parsed.content;
    resp["tool_calls"]    = tool_calls;
    resp["finish_reason"] = finish_reason;
    resp["usage"] = {
        {"prompt_tokens",     pt},
        {"completion_tokens", ct},
        {"total_tokens",      pt + ct},
    };

    emit(chat, "infer_success");
    const char* out = dup_cstr(resp.dump());
    if (!out) return build_error("INTERNAL_BRIDGE_ERROR", "failed to build response");
    return out;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

extern "C" llb_chat_t* llb_chat_create(const char* gguf_path,
                                       llb_event_cb event_cb,
                                       void* user_data) {
    if (!gguf_path) {
        emit_raw(event_cb, user_data, "create_failure:null_path");
        return nullptr;
    }

    // Existence probe so we fail fast with a sensible event.
    {
        FILE* f = fopen(gguf_path, "rb");
        if (!f) {
            emit_raw(event_cb, user_data, "create_failure:model_not_found");
            return nullptr;
        }
        fclose(f);
    }

    llb_chat_t* chat = new (std::nothrow) llb_chat();
    if (!chat) {
        emit_raw(event_cb, user_data, "create_failure:oom");
        return nullptr;
    }
    chat->model_path = gguf_path;
    chat->event_cb   = event_cb;
    chat->user_data  = user_data;

    emit(chat, "create_start");

    ensure_log_sink();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    // CPU-only build for an x86_64 Mac without a Metal-capable GPU.
    mparams.n_gpu_layers = 0;

    chat->model = llama_model_load_from_file(gguf_path, mparams);
    if (!chat->model) {
        emit(chat, "create_failure:load_model");
        delete chat;
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 4096;
    cparams.n_batch = 512;

    chat->ctx = llama_init_from_model(chat->model, cparams);
    if (!chat->ctx) {
        emit(chat, "create_failure:init_context");
        llama_model_free(chat->model);
        delete chat;
        return nullptr;
    }

    // Parse the model's chat template once (jinja). Empty override => use the
    // template embedded in the GGUF.
    try {
        chat->templates = common_chat_templates_init(chat->model, "");
    } catch (const std::exception& e) {
        emit(chat, "create_failure:chat_template");
        llama_free(chat->ctx);
        llama_model_free(chat->model);
        delete chat;
        return nullptr;
    }
    if (!chat->templates) {
        emit(chat, "create_failure:chat_template");
        llama_free(chat->ctx);
        llama_model_free(chat->model);
        delete chat;
        return nullptr;
    }

    // Hard gate: mochallama only loads tool-capable models. Compute caps from
    // the tool_use template variant when present (else the default), and reject
    // any model whose template cannot describe tools / round-trip tool calls.
    {
        tool_caps tc = compute_tool_caps(chat->model, chat->templates.get());
        if (!(tc.supports_tools && tc.supports_tool_calls)) {
            emit(chat, "create_failure:tools_unsupported");
            chat->templates.reset();
            llama_free(chat->ctx);
            llama_model_free(chat->model);
            delete chat;
            return nullptr;
        }
    }

    emit(chat, "create_success");
    return chat;
}

// Build the model-info JSON, optionally carrying an error reason. Never NULL.
static const char* build_model_info(const tool_caps& tc,
                                    const std::string& chat_format,
                                    const char* error) {
    json j;
    j["supports_tools"]        = tc.supports_tools;
    j["supports_tool_calls"]   = tc.supports_tool_calls;
    j["has_tool_use_template"] = tc.has_tool_use_template;
    if (error) {
        j["chat_format"] = nullptr;
        j["error"]       = error;
    } else {
        j["chat_format"] = chat_format;
        j["error"]       = nullptr;
    }
    const char* out = dup_cstr(j.dump());
    if (out) return out;
    // Last-ditch fallback if dup_cstr OOMs.
    static const char fallback[] =
        "{\"supports_tools\":false,\"supports_tool_calls\":false,"
        "\"has_tool_use_template\":false,\"chat_format\":null,"
        "\"error\":\"oom\"}";
    return dup_cstr(std::string(fallback));
}

extern "C" const char* llb_model_info(const char* gguf_path) {
    ensure_log_sink();
    tool_caps tc;  // defaults: all false

    if (!gguf_path) {
        return build_model_info(tc, "", "null_path");
    }
    {
        FILE* f = fopen(gguf_path, "rb");
        if (!f) return build_model_info(tc, "", "model_not_found");
        fclose(f);
    }

    llama_backend_init();

    // Load just the model (no inference context) — enough to read GGUF KV and
    // build chat templates. Cheaper than a full create; freed before returning.
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    struct llama_model* model = llama_model_load_from_file(gguf_path, mparams);
    if (!model) {
        return build_model_info(tc, "", "load_model");
    }

    common_chat_templates_ptr tmpls;
    try {
        tmpls = common_chat_templates_init(model, "");
    } catch (const std::exception&) {
        llama_model_free(model);
        return build_model_info(tc, "", "chat_template");
    }
    if (!tmpls) {
        llama_model_free(model);
        return build_model_info(tc, "", "chat_template");
    }

    tc = compute_tool_caps(model, tmpls.get());
    std::string fmt = probe_chat_format(tmpls.get());

    tmpls.reset();
    llama_model_free(model);

    return build_model_info(tc, fmt, nullptr);
}

extern "C" const char* llb_chat_infer(llb_chat_t* chat, const char* request_json) {
    return infer_impl(chat, request_json, nullptr, nullptr);
}

extern "C" const char* llb_chat_infer_stream(llb_chat_t* chat, const char* request_json,
                                             llb_token_cb token_cb, void* user_data) {
    return infer_impl(chat, request_json, token_cb, user_data);
}

/* ================================================================== */
/* LoRA adapters                                                       */
/* ================================================================== */

/* Push the current slot set onto the context.

   llama_set_adapters_lora replaces the whole set rather than adding to it, so
   every mutation rebuilds the arrays and reapplies. That is also why removing an
   adapter cannot simply "unset" one -- there is no such operation. */
static int apply_loras(llb_chat_t* chat) {
    std::vector<struct llama_adapter_lora*> adapters;
    std::vector<float>                      scales;
    adapters.reserve(chat->loras.size());
    scales.reserve(chat->loras.size());
    for (const auto& slot : chat->loras) {
        adapters.push_back(slot.adapter);
        scales.push_back(slot.scale);
    }
    return llama_set_adapters_lora(chat->ctx,
                                   adapters.empty() ? nullptr : adapters.data(),
                                   adapters.size(),
                                   scales.empty() ? nullptr : scales.data());
}

/* The adapter set, always returned in full after any operation, so a caller
   never has to model state the core already holds. */
static json lora_state(const llb_chat_t* chat) {
    json arr = json::array();
    for (const auto& slot : chat->loras) {
        arr.push_back({ {"id", slot.id}, {"path", slot.path}, {"scale", slot.scale} });
    }
    return arr;
}

extern "C" const char* llb_chat_lora(llb_chat_t* chat, const char* request_json) {
    if (!chat)         return build_error("ENGINE_NULL", "chat handle is null");
    if (!chat->ctx)    return build_error("ENGINE_CLOSED", "engine has no context");
    if (!request_json) return build_error("BAD_REQUEST", "request JSON is null");

    json req;
    try {
        req = json::parse(request_json);
    } catch (const std::exception& e) {
        return build_error("BAD_REQUEST", std::string("could not parse request JSON: ") + e.what());
    }

    const std::string op = req.value("op", "");
    if (op.empty()) {
        return build_error("BAD_REQUEST", "missing \"op\" (load|set|remove|clear|list)");
    }

    json out;
    out["type"] = "lora";

    if (op == "list") {
        out["adapters"] = lora_state(chat);
        return dup_cstr(out.dump());
    }

    if (op == "load") {
        const std::string path = req.value("path", "");
        if (path.empty()) return build_error("BAD_REQUEST", "\"load\" requires a \"path\"");

        struct llama_adapter_lora* adapter = llama_adapter_lora_init(chat->model, path.c_str());
        if (!adapter) {
            return build_error("LORA_LOAD_FAILED", "could not load LoRA adapter: " + path);
        }

        llb_lora_slot slot;
        slot.id      = chat->next_lora_id++;
        slot.path    = path;
        slot.scale   = req.value("scale", 1.0f);
        slot.adapter = adapter;
        chat->loras.push_back(slot);

        if (apply_loras(chat) != 0) {
            /* Roll back rather than leave the slot list describing a state the
               context is not actually in. */
            llama_adapter_lora_free(adapter);
            chat->loras.pop_back();
            apply_loras(chat);
            return build_error("LORA_APPLY_FAILED", "adapter loaded but could not be applied: " + path);
        }
        emit(chat, "lora_loaded");
        out["id"]       = slot.id;
        out["adapters"] = lora_state(chat);
        return dup_cstr(out.dump());
    }

    if (op == "set") {
        if (!req.contains("id")) return build_error("BAD_REQUEST", "\"set\" requires an \"id\"");
        const int   id    = req.value("id", -1);
        const float scale = req.value("scale", 1.0f);
        bool found = false;
        for (auto& slot : chat->loras) {
            if (slot.id == id) { slot.scale = scale; found = true; break; }
        }
        if (!found) return build_error("LORA_NOT_FOUND", "no adapter with id " + std::to_string(id));
        if (apply_loras(chat) != 0) {
            return build_error("LORA_APPLY_FAILED", "could not apply scale to adapter " + std::to_string(id));
        }
        out["id"]       = id;
        out["adapters"] = lora_state(chat);
        return dup_cstr(out.dump());
    }

    if (op == "remove") {
        if (!req.contains("id")) return build_error("BAD_REQUEST", "\"remove\" requires an \"id\"");
        const int id = req.value("id", -1);
        bool found = false;
        for (size_t i = 0; i < chat->loras.size(); ++i) {
            if (chat->loras[i].id == id) {
                llama_adapter_lora_free(chat->loras[i].adapter);
                chat->loras.erase(chat->loras.begin() + (long)i);
                found = true;
                break;
            }
        }
        if (!found) return build_error("LORA_NOT_FOUND", "no adapter with id " + std::to_string(id));
        apply_loras(chat);
        out["adapters"] = lora_state(chat);
        return dup_cstr(out.dump());
    }

    if (op == "clear") {
        for (auto& slot : chat->loras) llama_adapter_lora_free(slot.adapter);
        chat->loras.clear();
        apply_loras(chat);
        out["adapters"] = lora_state(chat);
        return dup_cstr(out.dump());
    }

    return build_error("BAD_REQUEST", "unknown op \"" + op + "\" (load|set|remove|clear|list)");
}

/* ================================================================== */
/* Embeddings and reranking                                            */
/* ================================================================== */

static enum llama_pooling_type parse_pooling(const std::string& s) {
    if (s == "none") return LLAMA_POOLING_TYPE_NONE;
    if (s == "mean") return LLAMA_POOLING_TYPE_MEAN;
    if (s == "cls")  return LLAMA_POOLING_TYPE_CLS;
    if (s == "last") return LLAMA_POOLING_TYPE_LAST;
    if (s == "rank") return LLAMA_POOLING_TYPE_RANK;
    return LLAMA_POOLING_TYPE_UNSPECIFIED;
}

extern "C" llb_embed_t* llb_embed_create(const char* gguf_path,
                                         const char* config_json,
                                         llb_event_cb event_cb,
                                         void* user_data) {
    if (!gguf_path) {
        emit_raw(event_cb, user_data, "create_failure:null_path");
        return nullptr;
    }
    {
        FILE* f = fopen(gguf_path, "rb");
        if (!f) {
            emit_raw(event_cb, user_data, "create_failure:model_not_found");
            return nullptr;
        }
        fclose(f);
    }

    json cfg = json::object();
    if (config_json && *config_json) {
        try {
            cfg = json::parse(config_json);
        } catch (const std::exception&) {
            emit_raw(event_cb, user_data, "create_failure:bad_config");
            return nullptr;
        }
    }

    /* `struct` tag required: the function llb_embed() shadows the struct name here. */
    llb_embed_t* e = new (std::nothrow) struct llb_embed();
    if (!e) {
        emit_raw(event_cb, user_data, "create_failure:oom");
        return nullptr;
    }
    e->model_path = gguf_path;
    e->event_cb   = event_cb;
    e->user_data  = user_data;
    e->pooling    = parse_pooling(cfg.value("pooling", std::string()));

    emit_raw(event_cb, user_data, "create_start");

    ensure_log_sink();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    e->model = llama_model_load_from_file(gguf_path, mparams);
    if (!e->model) {
        emit_raw(event_cb, user_data, "create_failure:load_model");
        delete e;
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    const int n_ctx = cfg.value("n_ctx", 0);
    e->n_batch      = cfg.value("n_batch", 512);
    e->n_seq_max    = cfg.value("n_seq_max", 8);
    if (n_ctx > 0) cparams.n_ctx = (uint32_t)n_ctx;
    cparams.n_batch      = (uint32_t)e->n_batch;
    /* An embedding context must also be told its batch can hold a whole sequence
       at once: unlike generation there is no incremental decode to fall back on. */
    cparams.n_ubatch     = (uint32_t)e->n_batch;
    cparams.n_seq_max    = (uint32_t)(e->n_seq_max > 0 ? e->n_seq_max : 1);
    cparams.embeddings   = true;
    cparams.pooling_type = e->pooling;

    e->ctx = llama_init_from_model(e->model, cparams);
    if (!e->ctx) {
        emit_raw(event_cb, user_data, "create_failure:init_context");
        llama_model_free(e->model);
        delete e;
        return nullptr;
    }
    llama_set_embeddings(e->ctx, true);
    /* Record what the context actually chose -- an UNSPECIFIED request resolves to
       the model's own default, and llb_rerank has to check the real value. */
    e->pooling = llama_pooling_type(e->ctx);

    emit_raw(event_cb, user_data, "create_success");
    return e;
}

/* Decode a group of token sequences in ONE batch and copy out their pooled
   embeddings.

   Each sequence gets its own llama_seq_id, so results map back by position with no
   bookkeeping. Packing several sequences per decode is what makes embedding a corpus
   practical -- one decode per text spends most of its time in setup rather than
   arithmetic.

   n_batch bounds the TOTAL tokens in one decode, so the caller chunks accordingly;
   n_seq_max bounds how many sequences a context will accept at once. */
static bool embed_group(llb_embed_t* e,
                        const std::vector<std::vector<llama_token>>& group,
                        int n_out,
                        std::vector<std::vector<float>>& out,
                        std::string& err) {
    if (group.empty()) return true;

    size_t total = 0;
    for (const auto& toks : group) {
        if (toks.empty()) { err = "empty token sequence"; return false; }
        total += toks.size();
    }
    if ((int)total > e->n_batch) {
        err = "batch of " + std::to_string(total) +
              " tokens exceeds n_batch " + std::to_string(e->n_batch);
        return false;
    }

    llama_memory_clear(llama_get_memory(e->ctx), true);

    llama_batch batch = llama_batch_init((int32_t)total, 0, (int32_t)group.size());
    common_batch_clear(batch);
    for (size_t seq = 0; seq < group.size(); ++seq) {
        const auto& toks = group[seq];
        for (size_t i = 0; i < toks.size(); ++i) {
            /* logits=true on every token: pooling reads all of them, and CLS/LAST
               pooling would otherwise read a position that was never computed. */
            common_batch_add(batch, toks[i], (llama_pos)i, { (llama_seq_id)seq }, true);
        }
    }

    const int rc = llama_decode(e->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        err = "llama_decode failed with code " + std::to_string(rc);
        return false;
    }

    for (size_t seq = 0; seq < group.size(); ++seq) {
        const float* emb = llama_get_embeddings_seq(e->ctx, (llama_seq_id)seq);
        if (!emb) {
            err = "no pooled embedding was produced (pooling type may be \"none\")";
            return false;
        }
        out.emplace_back(emb, emb + n_out);
    }
    return true;
}

/* Split a list of sequences into groups that each fit inside n_batch, then decode
   group by group. A single sequence longer than n_batch cannot be split further and
   is reported rather than silently truncated. */
static bool embed_all(llb_embed_t* e,
                      const std::vector<std::vector<llama_token>>& sequences,
                      int n_out,
                      std::vector<std::vector<float>>& out,
                      std::string& err) {
    const int  max_seqs = e->n_seq_max > 0 ? e->n_seq_max : 1;
    std::vector<std::vector<llama_token>> group;
    size_t group_tokens = 0;

    for (const auto& toks : sequences) {
        if ((int)toks.size() > e->n_batch) {
            err = "input of " + std::to_string(toks.size()) +
                  " tokens exceeds n_batch " + std::to_string(e->n_batch) +
                  "; raise n_batch or shorten the input";
            return false;
        }
        const bool would_overflow_tokens = (int)(group_tokens + toks.size()) > e->n_batch;
        const bool would_overflow_seqs   = (int)group.size() + 1 > max_seqs;
        if (!group.empty() && (would_overflow_tokens || would_overflow_seqs)) {
            if (!embed_group(e, group, n_out, out, err)) return false;
            group.clear();
            group_tokens = 0;
        }
        group_tokens += toks.size();
        group.push_back(toks);
    }
    return embed_group(e, group, n_out, out, err);
}

extern "C" const char* llb_embed(llb_embed_t* e, const char* request_json) {
    if (!e)            return build_error("ENGINE_NULL", "embed handle is null");
    if (!e->ctx)       return build_error("ENGINE_CLOSED", "engine has no context");
    if (!request_json) return build_error("BAD_REQUEST", "request JSON is null");

    json req;
    try {
        req = json::parse(request_json);
    } catch (const std::exception& ex) {
        return build_error("BAD_REQUEST", std::string("could not parse request JSON: ") + ex.what());
    }

    std::vector<std::string> inputs;
    if (req.contains("input")) {
        const json& in = req["input"];
        if (in.is_string())      inputs.push_back(in.get<std::string>());
        else if (in.is_array())  for (const auto& v : in) if (v.is_string()) inputs.push_back(v.get<std::string>());
    }
    if (inputs.empty()) {
        return build_error("BAD_REQUEST", "\"input\" must be a string or a non-empty array of strings");
    }

    if (e->pooling == LLAMA_POOLING_TYPE_NONE) {
        return build_error("POOLING_NONE",
            "this engine was created with pooling \"none\", so there is no per-input "
            "embedding to return; create it with \"mean\", \"cls\" or \"last\"");
    }

    const bool normalize = req.value("normalize", true);
    const int  n_embd    = llama_model_n_embd(e->model);

    std::vector<std::vector<llama_token>> sequences;
    sequences.reserve(inputs.size());
    int total_tokens = 0;
    for (const auto& text : inputs) {
        sequences.push_back(common_tokenize(e->ctx, text, true, true));
        total_tokens += (int)sequences.back().size();
    }

    std::vector<std::vector<float>> raw;
    raw.reserve(inputs.size());
    std::string err;
    if (!embed_all(e, sequences, n_embd, raw, err)) {
        return build_error("EMBED_FAILED", err);
    }

    json vectors = json::array();
    for (auto& vec : raw) {
        if (normalize) {
            std::vector<float> normed(vec.size());
            common_embd_normalize(vec.data(), normed.data(), (int)vec.size(), 2);
            vec.swap(normed);
        }
        vectors.push_back(vec);
    }

    json out;
    out["type"]       = "embedding";
    out["dim"]        = n_embd;
    out["embeddings"] = vectors;
    out["usage"]      = { {"prompt_tokens", total_tokens}, {"total_tokens", total_tokens} };
    return dup_cstr(out.dump());
}

extern "C" const char* llb_rerank(llb_embed_t* e, const char* request_json) {
    if (!e)            return build_error("ENGINE_NULL", "embed handle is null");
    if (!e->ctx)       return build_error("ENGINE_CLOSED", "engine has no context");
    if (!request_json) return build_error("BAD_REQUEST", "request JSON is null");

    if (e->pooling != LLAMA_POOLING_TYPE_RANK) {
        /* Refuse rather than return numbers that look like scores and are not:
           without RANK pooling the classification head is not in the graph at all. */
        return build_error("POOLING_NOT_RANK",
            "reranking requires a reranker model loaded with \"pooling\":\"rank\"");
    }

    json req;
    try {
        req = json::parse(request_json);
    } catch (const std::exception& ex) {
        return build_error("BAD_REQUEST", std::string("could not parse request JSON: ") + ex.what());
    }

    const std::string query = req.value("query", "");
    if (query.empty()) return build_error("BAD_REQUEST", "\"query\" is required");

    std::vector<std::string> docs;
    if (req.contains("documents") && req["documents"].is_array()) {
        for (const auto& v : req["documents"]) if (v.is_string()) docs.push_back(v.get<std::string>());
    }
    if (docs.empty()) return build_error("BAD_REQUEST", "\"documents\" must be a non-empty array of strings");

    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);
    const llama_token bos = llama_vocab_bos(vocab);
    const llama_token eos = llama_vocab_eos(vocab);
    const llama_token sep = llama_vocab_sep(vocab);

    const std::vector<llama_token> q_tokens = common_tokenize(e->ctx, query, false, true);

    struct scored { int index; float score; };
    std::vector<scored> results;
    int total_tokens = 0;

    for (size_t i = 0; i < docs.size(); ++i) {
        /* A reranker scores a PAIR, encoded as one sequence:
           [BOS] query [EOS] [SEP] document [EOS]
           This is the layout llama.cpp's own reranking path uses; feeding the two
           texts separately produces a number, just not a meaningful one. */
        std::vector<llama_token> d_tokens = common_tokenize(e->ctx, docs[i], false, true);
        std::vector<llama_token> pair;
        pair.reserve(q_tokens.size() + d_tokens.size() + 4);
        if (bos != LLAMA_TOKEN_NULL) pair.push_back(bos);
        pair.insert(pair.end(), q_tokens.begin(), q_tokens.end());
        if (eos != LLAMA_TOKEN_NULL) pair.push_back(eos);
        if (sep != LLAMA_TOKEN_NULL) pair.push_back(sep);
        pair.insert(pair.end(), d_tokens.begin(), d_tokens.end());
        if (eos != LLAMA_TOKEN_NULL) pair.push_back(eos);

        total_tokens += (int)pair.size();

        /* Under RANK pooling the sequence embedding is the classifier output --
           n_cls_out floats, normally one. */
        const int n_cls = (int)llama_model_n_cls_out(e->model);
        std::vector<std::vector<float>> scored_out;
        std::string err;
        if (!embed_group(e, { pair }, n_cls > 0 ? n_cls : 1, scored_out, err)) {
            return build_error("RERANK_FAILED", err);
        }
        const float value = (scored_out.empty() || scored_out[0].empty()) ? 0.0f : scored_out[0][0];
        results.push_back({ (int)i, value });
    }

    std::sort(results.begin(), results.end(),
              [](const scored& a, const scored& b) { return a.score > b.score; });

    size_t keep = results.size();
    if (req.contains("top_n")) {
        const int n = req.value("top_n", (int)results.size());
        if (n >= 0 && (size_t)n < keep) keep = (size_t)n;
    }

    json arr = json::array();
    for (size_t i = 0; i < keep; ++i) {
        /* The ORIGINAL index travels with the score: results are reordered, and a
           caller has to be able to map back to its own list. */
        arr.push_back({ {"index", results[i].index}, {"score", results[i].score} });
    }

    json out;
    out["type"]    = "rerank";
    out["results"] = arr;
    out["usage"]   = { {"prompt_tokens", total_tokens}, {"total_tokens", total_tokens} };
    return dup_cstr(out.dump());
}

extern "C" void llb_embed_destroy(llb_embed_t* e) {
    if (!e) return;
    if (e->ctx)   llama_free(e->ctx);
    if (e->model) llama_model_free(e->model);
    delete e;
}

extern "C" void llb_string_free(const char* s) {
    free((void*)s);
}

extern "C" void llb_chat_destroy(llb_chat_t* chat) {
    if (!chat) return;
    chat->closed = true;
    emit(chat, "destroy");
    chat->templates.reset();
    if (chat->ctx)   { llama_free(chat->ctx);         chat->ctx   = nullptr; }
    if (chat->model) { llama_model_free(chat->model); chat->model = nullptr; }
    delete chat;
}

extern "C" const char* llb_version(void) {
    return LLB_VERSION_STR;
}
