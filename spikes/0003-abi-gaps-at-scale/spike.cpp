// Spike 0003 — the four ABI gaps, measured.
//
// Throwaway. Links the same llama.cpp the bridge links, and answers four
// factual questions the ABI pass depends on. Nothing here is promoted; the
// numbers and the verdicts are the output.
//
//   spike schema  <model.gguf>   Q1  can a USER json schema constrain output?
//   spike prefix  <model.gguf>   Q2  does KV prefix reuse hold over a long agent run?
//   spike tokens  <model.gguf>   Q3  can we count tokens for a message list, cheaply?
//   spike cancel  <model.gguf>   Q4  can generation be stopped, and is the cache still sane?
//   spike slots   <model.gguf>   Q5  do N sequences share one context (the scaling shape)?

#include "llama.h"
#include "common.h"
#include "chat.h"
#include "sampling.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct Rig {
    llama_model*              model = nullptr;
    llama_context*            ctx   = nullptr;
    common_chat_templates_ptr tmpls;
    const llama_vocab*        vocab = nullptr;
};

static bool open_rig(Rig& r, const char* path, int n_ctx, int n_seq_max) {
    llama_backend_init();

    auto mp = llama_model_default_params();
    // Offload to the GPU, matching what the bridge now does by default. The
    // original run of this spike used llama's default of 0, so every number it
    // produced was a CPU number -- including the 9x headline.
    mp.n_gpu_layers = 999;
    r.model = llama_model_load_from_file(path, mp);
    if (!r.model) { fprintf(stderr, "FAIL: could not load %s\n", path); return false; }

    auto cp     = llama_context_default_params();
    cp.n_ctx    = n_ctx;
    cp.n_batch  = 512;
    cp.n_seq_max = n_seq_max;
    r.ctx = llama_init_from_model(r.model, cp);
    if (!r.ctx) { fprintf(stderr, "FAIL: could not create context\n"); return false; }

    r.tmpls = common_chat_templates_init(r.model, "");
    r.vocab = llama_model_get_vocab(r.model);
    return true;
}

static void close_rig(Rig& r) {
    if (r.ctx)   llama_free(r.ctx);
    if (r.model) llama_model_free(r.model);
    llama_backend_free();
}

static common_chat_msg msg(const char* role, const std::string& text) {
    common_chat_msg m;
    m.role    = role;
    m.content = text;
    return m;
}

// Apply the chat template. json_schema / grammar are the fields the bridge
// never sets — that omission IS gap Q1.
static common_chat_params apply_tmpl(Rig& r,
                                const std::vector<common_chat_msg>& messages,
                                const std::string& json_schema = "",
                                const std::string& gbnf        = "") {
    common_chat_templates_inputs in;
    in.messages              = messages;
    in.add_generation_prompt = true;
    in.use_jinja             = true;
    in.json_schema           = json_schema;
    in.grammar               = gbnf;
    return common_chat_templates_apply(r.tmpls.get(), in);
}

// Decode a token span into sequence `seq` starting at `pos0`. Returns false on
// a decode error. Uses the explicit batch API rather than llama_batch_get_one so
// the sequence id and start position are ours to choose — which is what both
// prefix reuse (Q2) and multi-slot (Q5) need.
static bool decode_span(Rig& r, const std::vector<llama_token>& toks,
                        size_t from, size_t to, llama_seq_id seq, int pos0,
                        bool logits_on_last) {
    const int n_batch = (int) llama_n_batch(r.ctx);
    llama_batch b = llama_batch_init(n_batch, 0, 1);
    bool ok = true;

    for (size_t i = from; i < to && ok; ) {
        size_t chunk = std::min((size_t) n_batch, to - i);
        b.n_tokens = 0;
        for (size_t k = 0; k < chunk; k++) {
            const size_t idx = i + k;
            b.token[b.n_tokens]     = toks[idx];
            b.pos[b.n_tokens]       = pos0 + (int) idx;
            b.n_seq_id[b.n_tokens]  = 1;
            b.seq_id[b.n_tokens][0] = seq;
            b.logits[b.n_tokens]    = (logits_on_last && idx == to - 1) ? 1 : 0;
            b.n_tokens++;
        }
        if (llama_decode(r.ctx, b) != 0) ok = false;
        i += chunk;
    }
    llama_batch_free(b);
    // llama_decode is ASYNC on Metal/CUDA. Without this, every timing below
    // measures enqueue latency, not work — which is how a prefix-reuse
    // benchmark can come back looking like noise.
    llama_synchronize(r.ctx);
    return ok;
}

// ------------------------------------------------------------------ Q1: schema

static int q_schema(Rig& r) {
    printf("\n== Q1  user-supplied JSON schema -> constrained output ==\n");

    const std::string schema = R"({
      "type": "object",
      "properties": {
        "city":       {"type": "string"},
        "population": {"type": "integer"},
        "founded":    {"type": "integer"}
      },
      "required": ["city", "population", "founded"],
      "additionalProperties": false
    })";

    std::vector<common_chat_msg> ms = {
        msg("user", "Tell me about Paris. Reply with city, population and the year it was founded.")
    };

    // Baseline: no schema. What does the model do unprompted?
    // Constrained: same prompt, schema attached.
    for (int pass = 0; pass < 2; pass++) {
        const bool constrained = (pass == 1);
        auto cp = apply_tmpl(r, ms, constrained ? schema : "");

        if (constrained) {
            printf("   grammar produced by common_chat_templates_apply: %zu bytes"
                   "  lazy=%d  triggers=%zu  preserved=%zu\n",
                   cp.grammar.size(), (int) cp.grammar_lazy,
                   cp.grammar_triggers.size(), cp.preserved_tokens.size());
            printf("   chat format: %s\n", common_chat_format_name(cp.format));
            size_t rp = cp.grammar.find("root ::=");
            printf("   root rule: %s\n",
                   rp == std::string::npos ? "(NO root RULE)"
                                           : cp.grammar.substr(rp, 160).c_str());
            printf("   generation_prompt: %s\n",
                   cp.generation_prompt.empty() ? "(empty)" : cp.generation_prompt.c_str());
            if (cp.grammar.empty()) {
                printf("   VERDICT Q1: NO — inputs.json_schema produced no grammar.\n");
                return 1;
            }
        }

        auto toks = common_tokenize(r.vocab, cp.prompt, true, true);
        llama_memory_clear(llama_get_memory(r.ctx), true);
        if (!decode_span(r, toks, 0, toks.size(), 0, 0, true)) return 1;

        common_params_sampling sp;
        sp.seed = 42;
        sp.temp = 0.7;
        if (!cp.grammar.empty()) {
            // TRAP: the bridge hardcodes TOOL_CALLS here. A schema grammar is
            // OUTPUT_FORMAT; a raw user GBNF is USER — and the three differ in
            // whether the generation prompt is prefilled into the sampler
            // (common_grammar_needs_prefill).
            sp.grammar = { COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, cp.grammar };
        }
        sp.grammar_lazy      = cp.grammar_lazy;
        sp.grammar_triggers  = cp.grammar_triggers;
        sp.generation_prompt = cp.generation_prompt;

        common_sampler* smpl = common_sampler_init(r.model, sp);
        if (!smpl) { printf("   FAIL: sampler init failed\n"); return 1; }

        std::string out;
        int pos = (int) toks.size();
        for (int i = 0; i < 160; i++) {
            llama_token t = common_sampler_sample(smpl, r.ctx, -1);
            common_sampler_accept(smpl, t, true);
            if (llama_vocab_is_eog(r.vocab, t)) break;
            out += common_token_to_piece(r.vocab, t, false);
            std::vector<llama_token> one{t};
            if (!decode_span(r, one, 0, 1, 0, pos++, true)) break;
        }
        common_sampler_free(smpl);

        printf("   %-12s %s\n", constrained ? "CONSTRAINED" : "baseline", out.c_str());

        if (constrained) {
            // The grammar's own root rule is
            //   root ::= "<|im_start|>assistant\n" space space ("```json" space RF space "```" | RF)
            // i.e. upstream DELIBERATELY allows a markdown fence. So conformant
            // output is not necessarily parseable JSON, and a core that returns
            // the raw text hands the caller a string it cannot json.loads().
            // Strip the fence the way common_chat_parse would.
            std::string body = out;
            size_t f = body.find("```json");
            if (f != std::string::npos) {
                body = body.substr(f + 7);
                size_t e = body.rfind("```");
                if (e != std::string::npos) body = body.substr(0, e);
            }
            printf("   fenced: %s\n", f != std::string::npos ? "YES (must be stripped by the core)" : "no");

            bool valid = false;
            try {
                auto parsed = json::parse(body);
                valid = parsed.contains("city") && parsed.contains("population")
                     && parsed.contains("founded") && parsed.size() == 3
                     && parsed["population"].is_number_integer();
            } catch (...) { valid = false; }
            printf("   parses + matches schema exactly: %s\n", valid ? "YES" : "NO");
            printf("   VERDICT Q1: %s\n",
                   valid ? "YES — inputs.json_schema is all we need; the bridge simply never sets it.\n"
                           "               TRAP: the grammar permits a ```json fence, so the core MUST parse\n"
                           "               before returning, or callers get a string that is not JSON."
                         : "NO — schema attached but output did not conform.");
            return valid ? 0 : 1;
        }
    }
    return 0;
}

// ------------------------------------------------------------------ Q2: prefix

// Longest common prefix between the cached token sequence and the new one.
static size_t common_prefix(const std::vector<llama_token>& a,
                            const std::vector<llama_token>& b) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) n++;
    return n;
}

static int q_prefix(Rig& r) {
    printf("\n== Q2  KV prefix reuse across a growing agent conversation ==\n");
    printf("   %-4s %10s %10s %10s %10s %8s\n",
           "turn", "tok_total", "clear_ms", "reuse_ms", "reused", "speedup");

    std::vector<common_chat_msg> ms = {
        msg("system", "You are a terse assistant with access to tools.")
    };

    // Warm-up: the first decode on Metal compiles graphs. Measuring it would
    // charge turn 1 for the backend's start-up.
    {
        auto warm = common_tokenize(r.vocab, std::string("warm up the backend"), true, true);
        llama_memory_clear(llama_get_memory(r.ctx), true);
        decode_span(r, warm, 0, warm.size(), 0, 0, true);
    }

    std::vector<llama_token> cached;   // what sequence 0 currently holds
    double clear_total = 0, reuse_total = 0;
    size_t last_total_tokens = 0;

    const int TURNS = 32;
    for (int turn = 1; turn <= TURNS; turn++) {
        // A realistic agent turn: user asks, assistant answers, a tool result
        // lands. The prompt grows monotonically and its prefix never changes —
        // the exact shape toolnexus's loop produces.
        ms.push_back(msg("user", "Step " + std::to_string(turn) +
                                 ": look up record " + std::to_string(1000 + turn) +
                                 " and summarise the delta against the previous one."));
        auto cp   = apply_tmpl(r, ms);
        auto toks = common_tokenize(r.vocab, cp.prompt, true, true);
        last_total_tokens = toks.size();

        // --- today's behaviour: clear, re-prefill everything ---------------
        llama_memory_clear(llama_get_memory(r.ctx), true);
        double t0 = now_ms();
        if (!decode_span(r, toks, 0, toks.size(), 0, 0, true)) { printf("   decode failed\n"); return 1; }
        double clear_ms = now_ms() - t0;

        // --- proposed: keep the cache, drop the divergent tail -------------
        // Rebuild the cache to `cached` state first so the measurement is honest.
        llama_memory_clear(llama_get_memory(r.ctx), true);
        if (!cached.empty()) {
            if (!decode_span(r, cached, 0, cached.size(), 0, 0, true)) return 1;
        }
        size_t reuse = common_prefix(cached, toks);
        double t1 = now_ms();
        llama_memory_seq_rm(llama_get_memory(r.ctx), 0, (llama_pos) reuse, -1);
        if (reuse < toks.size()) {
            if (!decode_span(r, toks, reuse, toks.size(), 0, 0, true)) { printf("   tail decode failed\n"); return 1; }
        }
        double reuse_ms = now_ms() - t1;

        cached = toks;
        clear_total += clear_ms;
        reuse_total += reuse_ms;

        if (turn % 4 == 0 || turn == 1) {
            printf("   %-4d %10zu %10.1f %10.1f %10zu %7.1fx\n",
                   turn, toks.size(), clear_ms, reuse_ms, reuse,
                   reuse_ms > 0 ? clear_ms / reuse_ms : 0.0);
        }

        // The assistant reply and tool result, appended for the next turn.
        ms.push_back(msg("assistant", "Record " + std::to_string(1000 + turn) +
                                      " differs in three fields; see the summary."));
    }

    printf("\n   total prefill over %d turns:  clear=%.0f ms   reuse=%.0f ms   %.1fx\n",
           TURNS, clear_total, reuse_total,
           reuse_total > 0 ? clear_total / reuse_total : 0.0);
    printf("   final prompt: %zu tokens\n", last_total_tokens);
    printf("   VERDICT Q2: %s\n",
           (reuse_total * 2 < clear_total)
             ? "YES — reuse is materially cheaper and the gap widens with conversation length."
             : "NO — reuse did not pay for itself at this size.");
    return 0;
}

// ------------------------------------------------------------------ Q3: tokens

static int q_tokens(Rig& r) {
    printf("\n== Q3  counting tokens for a message list ==\n");

    std::vector<common_chat_msg> ms = { msg("system", "You are a terse assistant.") };
    for (int i = 0; i < 40; i++) {
        ms.push_back(msg("user", "Question " + std::to_string(i) + " about a moderately long topic that takes a sentence or two to state properly."));
        ms.push_back(msg("assistant", "A correspondingly wordy answer, of the sort a real assistant produces when it is being helpful rather than terse."));
    }

    double t0 = now_ms();
    auto cp   = apply_tmpl(r, ms);
    double t_tmpl = now_ms() - t0;

    double t1 = now_ms();
    auto toks = common_tokenize(r.vocab, cp.prompt, true, true);
    double t_tok = now_ms() - t1;

    printf("   %zu messages -> %zu tokens\n", ms.size(), toks.size());
    printf("   template apply: %.2f ms   tokenize: %.2f ms   total: %.2f ms\n",
           t_tmpl, t_tok, t_tmpl + t_tok);
    printf("   n_ctx = %u, so this prompt is %.0f%% of the window\n",
           llama_n_ctx(r.ctx), 100.0 * toks.size() / llama_n_ctx(r.ctx));
    printf("   VERDICT Q3: %s — counting needs the model's vocab + template, "
           "which only the core has. No context decode required.\n",
           (t_tmpl + t_tok) < 50 ? "YES, and it is cheap" : "YES, but not free");
    return 0;
}

// ------------------------------------------------------------------ Q4: cancel

static int q_cancel(Rig& r) {
    printf("\n== Q4  stopping generation mid-flight, and cache integrity after ==\n");

    std::vector<common_chat_msg> ms = { msg("user", "Count slowly from one to fifty in words, one per line.") };
    auto cp   = apply_tmpl(r, ms);
    auto toks = common_tokenize(r.vocab, cp.prompt, true, true);

    llama_memory_clear(llama_get_memory(r.ctx), true);
    if (!decode_span(r, toks, 0, toks.size(), 0, 0, true)) return 1;

    common_params_sampling sp;
    sp.seed = 7;
    sp.temp = 0.7;
    common_sampler* smpl = common_sampler_init(r.model, sp);

    // Abort at token 12, the way a cancelled context or a closed stream would.
    const int ABORT_AT = 12;
    std::string partial;
    int pos = (int) toks.size();
    int produced = 0;
    for (int i = 0; i < 400; i++) {
        if (produced >= ABORT_AT) break;              // <-- the callback says stop
        llama_token t = common_sampler_sample(smpl, r.ctx, -1);
        common_sampler_accept(smpl, t, true);
        if (llama_vocab_is_eog(r.vocab, t)) break;
        partial += common_token_to_piece(r.vocab, t, false);
        std::vector<llama_token> one{t};
        if (!decode_span(r, one, 0, 1, 0, pos++, true)) break;
        produced++;
    }
    common_sampler_free(smpl);
    printf("   aborted after %d tokens: %s\n", produced,
           partial.substr(0, 60).c_str());
    printf("   cache now holds %d positions (prompt %zu + %d generated)\n",
           pos, toks.size(), produced);

    // The interaction that actually matters: after an abort, is the retained
    // cache still usable? Roll back to the prompt and run a DIFFERENT request.
    llama_memory_seq_rm(llama_get_memory(r.ctx), 0, (llama_pos) toks.size(), -1);

    std::vector<common_chat_msg> ms2 = { msg("user", "Name the capital of France in one word.") };
    auto cp2   = apply_tmpl(r, ms2);
    auto toks2 = common_tokenize(r.vocab, cp2.prompt, true, true);
    size_t reuse = common_prefix(toks, toks2);
    llama_memory_seq_rm(llama_get_memory(r.ctx), 0, (llama_pos) reuse, -1);
    if (!decode_span(r, toks2, reuse, toks2.size(), 0, 0, true)) { printf("   FAIL: post-abort decode\n"); return 1; }

    common_sampler* s2 = common_sampler_init(r.model, sp);
    std::string after;
    int pos2 = (int) toks2.size();
    for (int i = 0; i < 24; i++) {
        llama_token t = common_sampler_sample(s2, r.ctx, -1);
        common_sampler_accept(s2, t, true);
        if (llama_vocab_is_eog(r.vocab, t)) break;
        after += common_token_to_piece(r.vocab, t, false);
        std::vector<llama_token> one{t};
        if (!decode_span(r, one, 0, 1, 0, pos2++, true)) break;
    }
    common_sampler_free(s2);

    printf("   next request after abort (reused %zu of %zu prompt tokens): %s\n",
           reuse, toks2.size(), after.c_str());
    bool sane = after.find("Paris") != std::string::npos;
    printf("   VERDICT Q4: %s\n",
           sane ? "YES — abort is a caller-side loop break; rolling the cache back leaves it usable."
                : "SUSPECT — output after abort does not look right; inspect before shipping.");
    return sane ? 0 : 1;
}

// ------------------------------------------------------------------ Q5: slots

static int q_slots(Rig& r, int n_slots) {
    printf("\n== Q5  N independent sequences in ONE context (the scaling shape) ==\n");
    printf("   n_seq_max = %u\n", llama_n_seq_max(r.ctx));

    // Each slot gets a DIFFERENT conversation, all resident at once. This is
    // what llama-server calls a slot; the current one-handle-one-conversation
    // design cannot express it.
    std::vector<std::vector<llama_token>> prompts;
    for (int s = 0; s < n_slots; s++) {
        std::vector<common_chat_msg> ms = {
            msg("user", "In one short sentence, what is interesting about the number " +
                        std::to_string(s * 7 + 3) + "?")
        };
        auto cp = apply_tmpl(r, ms);
        prompts.push_back(common_tokenize(r.vocab, cp.prompt, true, true));
    }

    common_params_sampling sp; sp.seed = 3; sp.temp = 0.7;
    const int STEPS = 24;

    // Run the same work twice: serially (one slot at a time, today's shape) and
    // batched (all slots in one decode, the server shape). Same total tokens.
    double serial_ms = 0, batched_ms = 0;
    std::vector<std::string> outs(n_slots);

    for (int mode = 0; mode < 2; mode++) {
        const bool batched = (mode == 1);
        llama_memory_clear(llama_get_memory(r.ctx), true);

        std::vector<common_sampler*> smpls;
        for (int s = 0; s < n_slots; s++) smpls.push_back(common_sampler_init(r.model, sp));
        std::vector<int> pos(n_slots);
        std::vector<llama_token> next(n_slots);
        std::vector<bool> done(n_slots, false);
        std::vector<std::string> got(n_slots);

        double t0 = now_ms();

        // Prefill every slot except each prompt's final token (no logits needed).
        for (int s = 0; s < n_slots; s++) {
            if (!decode_span(r, prompts[s], 0, prompts[s].size() - 1,
                             (llama_seq_id) s, 0, false)) {
                printf("   FAIL: slot %d prefill\n", s);
                return 1;
            }
            pos[s] = (int) prompts[s].size() - 1;
        }

        // THE FIX. Every slot's final prompt token goes into ONE batch with its
        // own logits row, and each sampler reads ITS index. Sampling with idx=-1
        // after per-slot decodes makes slot 0 read slot N-1's logits — which is
        // how four slots produce four streams of confident garbage while looking
        // "distinct". A per-slot design MUST carry the logits index.
        const int n_batch = (int) llama_n_batch(r.ctx);
        llama_batch b = llama_batch_init(n_batch, 0, 1);

        for (int step = 0; step <= STEPS; step++) {
            b.n_tokens = 0;
            std::vector<int> row_of(n_slots, -1);
            for (int s = 0; s < n_slots; s++) {
                if (done[s]) continue;
                llama_token t = (step == 0) ? prompts[s].back() : next[s];
                row_of[s] = b.n_tokens;
                b.token[b.n_tokens]     = t;
                b.pos[b.n_tokens]       = pos[s]++;
                b.n_seq_id[b.n_tokens]  = 1;
                b.seq_id[b.n_tokens][0] = (llama_seq_id) s;
                b.logits[b.n_tokens]    = 1;
                b.n_tokens++;

                if (!batched) {                 // serial: one decode per slot
                    if (llama_decode(r.ctx, b) != 0) { printf("   decode failed\n"); return 1; }
                    llama_synchronize(r.ctx);
                    llama_token nt = common_sampler_sample(smpls[s], r.ctx, 0);
                    common_sampler_accept(smpls[s], nt, true);
                    if (llama_vocab_is_eog(r.vocab, nt)) done[s] = true;
                    else { got[s] += common_token_to_piece(r.vocab, nt, false); next[s] = nt; }
                    b.n_tokens = 0;
                }
            }
            if (batched && b.n_tokens > 0) {    // batched: ONE decode for all slots
                if (llama_decode(r.ctx, b) != 0) { printf("   decode failed\n"); return 1; }
                llama_synchronize(r.ctx);
                for (int s = 0; s < n_slots; s++) {
                    if (row_of[s] < 0) continue;
                    llama_token nt = common_sampler_sample(smpls[s], r.ctx, row_of[s]);
                    common_sampler_accept(smpls[s], nt, true);
                    if (llama_vocab_is_eog(r.vocab, nt)) done[s] = true;
                    else { got[s] += common_token_to_piece(r.vocab, nt, false); next[s] = nt; }
                }
            }
        }
        llama_batch_free(b);
        for (auto* s : smpls) common_sampler_free(s);

        double ms = now_ms() - t0;
        if (batched) { batched_ms = ms; outs = got; } else { serial_ms = ms; }
    }

    for (int s = 0; s < n_slots; s++)
        printf("   slot %d: %s\n", s, outs[s].substr(0, 70).c_str());

    const double toks = (double) n_slots * STEPS;
    printf("\n   %d slots x %d tokens:  serial=%.0f ms (%.0f tok/s)   batched=%.0f ms (%.0f tok/s)   %.1fx\n",
           n_slots, STEPS, serial_ms, toks * 1000 / serial_ms,
           batched_ms, toks * 1000 / batched_ms, serial_ms / batched_ms);

    bool distinct = true, sane = true;
    for (int s = 1; s < n_slots; s++) if (outs[s] == outs[0]) distinct = false;
    for (int s = 0; s < n_slots; s++) if (outs[s].size() < 10) sane = false;

    printf("   VERDICT Q5: %s\n",
           (distinct && sane)
             ? "YES — one context holds N independent conversations, and batching them\n"
               "               into a single decode is where the throughput is. The scaling\n"
               "               shape is slots + per-slot logits index, available today."
             : "SUSPECT — slots did not produce independent, sane output.");
    return (distinct && sane) ? 0 : 1;
}


// ------------------------------------------------------------------ main

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: spike <schema|prefix|tokens|cancel|slots|all> <model.gguf>\n");
        return 2;
    }
    const std::string what  = argv[1];
    const char*       model = argv[2];

    // Quiet. llama.cpp's default logging drowns the measurements.
    llama_log_set([](ggml_log_level lvl, const char* text, void*) {
        if (lvl >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    }, nullptr);
    common_log_pause(common_log_main());

    const bool all   = (what == "all");
    const int  n_ctx = (what == "slots" || all) ? 8192 : 8192;
    const int  slots = 4;

    Rig r;
    if (!open_rig(r, model, n_ctx, (what == "slots" || all) ? slots : 1)) return 1;

    printf("model: %s\n", model);
    printf("n_ctx: %d   n_seq_max: %u\n", n_ctx, llama_n_seq_max(r.ctx));

    int rc = 0;
    if (all || what == "schema") rc |= q_schema(r);
    if (all || what == "tokens") rc |= q_tokens(r);
    if (all || what == "prefix") rc |= q_prefix(r);
    if (all || what == "cancel") rc |= q_cancel(r);
    if (all || what == "slots")  rc |= q_slots(r, slots);

    close_rig(r);
    printf("\n%s\n", rc == 0 ? "ALL VERDICTS POSITIVE" : "AT LEAST ONE VERDICT NEGATIVE — read above");
    return rc;
}
