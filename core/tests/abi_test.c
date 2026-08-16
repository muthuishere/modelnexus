/* C-level conformance for the inference-control surface (ADR-0008).
 *
 * Deliberately plain C with no test framework: this is the ABI, and the point
 * is that it is reachable with nothing but a C compiler and a dlopen.
 *
 * Skips (exit 0) when no model is available, rather than failing — a skipped
 * suite must not look like a passing one, so it says so loudly.
 *
 *   MODELNEXUS_MODEL=~/.chatbot_models/qwen2.5-1.5b-instruct-q4_k_m.gguf ./abi_test
 */
#include "llamabridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static int checks   = 0;

static void ok(int cond, const char* what) {
    checks++;
    if (cond) { printf("  ok    %s\n", what); }
    else      { printf("  FAIL  %s\n", what); failures++; }
}

/* Crude substring probe. Enough for a C-level smoke of a JSON contract; the
   binding suites assert on parsed structure. */
static int has(const char* hay, const char* needle) {
    return hay && strstr(hay, needle) != NULL;
}

/* ---------------------------------------------------------------- cancel */

static int cancel_after = 0;
static int tokens_seen  = 0;

static int counting_cb(const char* piece, void* user_data) {
    (void) piece; (void) user_data;
    tokens_seen++;
    return (cancel_after > 0 && tokens_seen >= cancel_after) ? 1 : 0;
}

int main(void) {
    const char* model = getenv("MODELNEXUS_MODEL");
    if (!model || !*model) {
        printf("SKIP: set MODELNEXUS_MODEL to a chat GGUF to run the ABI tests\n");
        return 0;
    }

    printf("llb_version: %s\n", llb_version());
    llb_set_log_level(0);   /* silence */

    /* --- create with a config (0.2.0 signature) ------------------------ */
    llb_chat_t* chat = llb_chat_create(model, "{\"n_ctx\":4096,\"n_batch\":512}", NULL, NULL);
    if (!chat) { printf("FAIL: could not create engine from %s\n", model); return 1; }
    printf("engine created with an explicit config\n");

    /* --- NULL config must behave exactly as before --------------------- */
    {
        llb_chat_t* c2 = llb_chat_create(model, NULL, NULL, NULL);
        ok(c2 != NULL, "NULL config is accepted and behaves as the old signature");
        if (c2) llb_chat_destroy(c2);
    }

    const char* CONV1 =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Name the capital of France in one word.\"}],"
        "\"max_tokens\":16,\"seed\":42,\"temperature\":0.0}";

    /* --- token counting ------------------------------------------------ */
    {
        const char* r = llb_count_tokens(chat, CONV1);
        ok(has(r, "\"type\":\"token_count\""), "llb_count_tokens returns a token_count");
        ok(has(r, "\"tokens\":") && has(r, "\"n_ctx\":"), "it reports tokens and n_ctx");
        printf("        %s\n", r ? r : "(null)");
        llb_string_free(r);
    }

    /* --- reuse on vs off must produce IDENTICAL text -------------------
       Reuse is a latency property. Any observable difference in output is a
       defect, and this is the assertion that catches it. */
    {
        const char* NOREUSE =
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Name the capital of France in one word.\"}],"
            "\"max_tokens\":16,\"seed\":42,\"temperature\":0.0,\"reuse_cache\":false}";

        const char* a = llb_chat_infer(chat, NOREUSE);   /* cold, cache cleared */
        const char* b = llb_chat_infer(chat, CONV1);     /* warm, prefix reused */
        const char* c = llb_chat_infer(chat, NOREUSE);   /* cold again */

        ok(a && b && c, "three inferences returned");
        ok(strcmp(a, b) == 0, "reuse_cache=true matches reuse_cache=false byte for byte");
        ok(strcmp(a, c) == 0, "repeated cold runs are stable (the control)");
        if (strcmp(a, b) != 0) {
            printf("        no-reuse: %s\n        reuse:    %s\n", a, b);
        }
        llb_string_free(a); llb_string_free(b); llb_string_free(c);
    }

    /* --- cancellation, and the cache is still sane afterwards -----------
       This is the D2xD4 interaction: an abort leaves a partial assistant turn
       in the cache, and without rollback the next call's prefix match extends
       a truncated turn as though it were complete. Silent, plausible, wrong. */
    {
        const char* LONG =
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Count slowly from one to fifty in words, one per line.\"}],"
            "\"max_tokens\":300,\"seed\":7,\"temperature\":0.0}";

        tokens_seen = 0; cancel_after = 8;
        const char* r = llb_chat_infer_stream(chat, LONG, counting_cb, NULL);
        ok(has(r, "\"finish_reason\":\"cancelled\""), "a non-zero callback return cancels");
        ok(tokens_seen == 8, "generation stopped at the requested token, not later");
        ok(has(r, "\"completion_tokens\":8"), "usage reports the tokens actually generated");
        llb_string_free(r);

        /* The proof that rollback happened: a DIFFERENT request on the same
           engine must be correct, uninfluenced by the abandoned partial turn. */
        cancel_after = 0;
        const char* after = llb_chat_infer(chat, CONV1);
        ok(has(after, "Paris"), "a request after a cancellation is still correct");
        if (!has(after, "Paris")) printf("        got: %s\n", after);
        llb_string_free(after);
    }

    /* --- callback returning 0 must not disturb anything ---------------- */
    {
        tokens_seen = 0; cancel_after = 0;
        const char* r = llb_chat_infer_stream(chat, CONV1, counting_cb, NULL);
        ok(!has(r, "\"finish_reason\":\"cancelled\""), "returning 0 lets generation finish");
        ok(tokens_seen > 0, "the callback saw tokens");
        llb_string_free(r);
    }

    /* --- cache: inspect and drop ---------------------------------------
       The assertion that matters is that "clear" is OBSERVABLE. A clear that
       silently did nothing would still return a well-formed response, and the
       next inference would still be correct — just slow, and still holding the
       previous tenant's conversation. So: infer, see a non-zero cache, clear,
       see zero, then infer again and prove the handle still works. */
    {
        const char* r0 = llb_chat_infer(chat, CONV1);
        llb_string_free(r0);

        const char* st = llb_chat_cache(chat, "{\"op\":\"status\"}");
        ok(has(st, "\"type\":\"cache\""), "llb_chat_cache reports status");
        ok(!has(st, "\"tokens\":0"), "after an inference the cache is not empty");
        printf("        before: %s\n", st ? st : "(null)");
        llb_string_free(st);

        const char* cl = llb_chat_cache(chat, "{\"op\":\"clear\"}");
        ok(has(cl, "\"tokens\":0"), "clear empties the cache, and says so");
        printf("        after:  %s\n", cl ? cl : "(null)");
        llb_string_free(cl);

        const char* st2 = llb_chat_cache(chat, "{\"op\":\"status\"}");
        ok(has(st2, "\"tokens\":0"), "the clear persisted — status agrees");
        llb_string_free(st2);

        const char* after = llb_chat_infer(chat, CONV1);
        ok(has(after, "Paris"), "the engine still works after a clear");
        llb_string_free(after);

        /* A default (empty) request is a status read, not a destructive op —
           getting that backwards would make an innocent-looking call wipe a
           conversation. */
        const char* def = llb_chat_cache(chat, NULL);
        ok(!has(def, "\"tokens\":0"), "a NULL request reads status, it does not clear");
        llb_string_free(def);

        const char* bad = llb_chat_cache(chat, "{\"op\":\"obliterate\"}");
        ok(has(bad, "INVALID_REQUEST"), "an unknown cache op is rejected, not ignored");
        llb_string_free(bad);
    }

    /* --- constrained output -------------------------------------------- */
    {
        const char* SCHEMA =
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Describe Paris.\"}],"
            "\"max_tokens\":120,\"seed\":42,\"temperature\":0.0,"
            "\"json_schema\":{\"type\":\"object\",\"properties\":{"
            "\"city\":{\"type\":\"string\"},\"country\":{\"type\":\"string\"}},"
            "\"required\":[\"city\",\"country\"],\"additionalProperties\":false}}";

        const char* r = llb_chat_infer(chat, SCHEMA);
        /* The response JSON escapes the content's quotes, so probe the bare
           key names rather than a quoted form that only exists unescaped. */
        ok(has(r, "city") && has(r, "country"),
           "schema-constrained output carries the schema's fields");
        /* The fence check: upstream's grammar PERMITS ```json, so a core that
           returns raw text hands back a string that is not JSON. */
        ok(!has(r, "```"), "the markdown fence was stripped by the core");
        printf("        %s\n", r ? r : "(null)");
        llb_string_free(r);
    }

    /* --- schema and grammar together is an error, not a precedence rule - */
    {
        const char* BOTH =
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
            "\"json_schema\":{\"type\":\"object\"},\"grammar\":\"root ::= \\\"x\\\"\"}";
        const char* r = llb_chat_infer(chat, BOTH);
        ok(has(r, "\"type\":\"error\"") && has(r, "INVALID_REQUEST"),
           "json_schema + grammar is rejected rather than silently resolved");
        llb_string_free(r);
    }

    /* --- a raw GBNF grammar constrains too ----------------------------- */
    {
        const char* GBNF =
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Pick a colour.\"}],"
            "\"max_tokens\":16,\"seed\":42,\"temperature\":0.0,"
            "\"grammar\":\"root ::= \\\"red\\\" | \\\"blue\\\"\"}";
        const char* r = llb_chat_infer(chat, GBNF);
        ok(has(r, "red") || has(r, "blue"), "a raw GBNF grammar constrains generation");
        printf("        %s\n", r ? r : "(null)");
        llb_string_free(r);
    }

    /* --- create failures are DATA, not a string in an event ------------
       The bindings used to string-match the event and invent their own codes.
       That is behaviour living in a binding, which is the thing this ABI
       exists to prevent. */
    {
        llb_chat_t* nope = llb_chat_create("/definitely/not/a/model.gguf", NULL, NULL, NULL);
        ok(nope == NULL, "a missing model still returns NULL");
        const char* why = llb_last_error();
        ok(has(why, "MODEL_NOT_FOUND"), "and llb_last_error says why, with a stable code");
        printf("        %s\n", why ? why : "(null)");
        llb_string_free(why);

        /* A successful create must clear it, or the next caller reads a stale
           reason and reports a failure that did not happen. */
        llb_chat_t* fine = llb_chat_create(model, NULL, NULL, NULL);
        ok(fine != NULL, "a good model still loads");
        ok(llb_last_error() == NULL, "a successful create clears the last error");
        if (fine) llb_chat_destroy(fine);
    }

    llb_chat_destroy(chat);

    printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
