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

struct llb_chat {
    std::string                model_path;
    llb_event_cb               event_cb  = nullptr;
    void*                      user_data = nullptr;
    bool                       closed    = false;
    struct llama_model*        model     = nullptr;
    struct llama_context*      ctx       = nullptr;
    common_chat_templates_ptr  templates;   // owns the parsed chat template(s)
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
