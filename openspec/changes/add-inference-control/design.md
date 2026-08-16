# Design — inference control

Everything here is grounded in [spike 0003](../../../spikes/0003-abi-gaps-at-scale/). Where a
choice was made against an alternative, the alternative is named.

## Shape of the request

Constrained output and cache control are **request fields**, not entry points. That is ADR-0002
paying off: a new generation parameter costs no symbol and no binding change.

```jsonc
{
  "messages": [ ... ],
  "tools":    [ ... ],

  // NEW — at most one of these two.
  "json_schema": { "type": "object", ... },   // schema  -> OUTPUT_FORMAT grammar
  "grammar":     "root ::= ...",              // raw GBNF -> USER grammar

  // NEW — cache control. Absent = reuse (the default).
  "reuse_cache": true
}
```

Supplying both `json_schema` and `grammar` is an error rather than a precedence rule. A silent
winner between two output constraints is a debugging session nobody should have.

### Why the grammar type is not cosmetic

`llamabridge.cpp:381` currently hardcodes `COMMON_GRAMMAR_TYPE_TOOL_CALLS` for whatever grammar
the template produced. Three sources, three types:

| source | type | prefilled? |
|---|---|---|
| chat template tool-calling | `TOOL_CALLS` | yes |
| `json_schema` | `OUTPUT_FORMAT` | yes |
| `grammar` (raw GBNF) | `USER` | **no** |

`common_grammar_needs_prefill` (`common/common.h:204`) returns false only for `USER`. Passing a
caller's GBNF as `TOOL_CALLS` prefills the generation prompt into a grammar that was never
written to accept it.

### The core parses; the caller does not

Spike 0003 found the schema grammar's root rule is

```
root ::= "<|im_start|>assistant\n" space space ("```json" space response-format space "```" | response-format)
```

Upstream **deliberately** permits a markdown fence, so this is conformant output:

````
```json
{ "city": "Paris", "population": 2, "founded": 256 }
```
````

If the core returns raw text, a caller who was promised "output matching your schema" gets a
string that fails `json.loads`. The core already runs `common_chat_parse` for tool calls; the
schema path uses the same parse, and the response carries the parsed content.

**Rejected:** documenting the fence and making callers strip it. It pushes an upstream detail
into five bindings and every consumer, and it makes the feature's headline claim false.

## KV prefix reuse

### Mechanism

`llb_chat` retains the token sequence currently resident in the cache. Per call:

1. Build the prompt and tokenize as today.
2. `n = longest_common_prefix(cached, new)`.
3. `llama_memory_seq_rm(mem, seq, n, -1)` — drop the divergent tail.
4. Decode `new[n..]` only.
5. Store `new` as the cached sequence.

`n == 0` degenerates to today's behaviour, so the worst case is what we already ship.

### Why default-on

An agent loop appends; its prefix never changes. That is the shape toolnexus produces and the
shape the measurement used. Default-off would mean the common case pays the quadratic cost
unless it knows to ask, and nobody knows to ask.

`"reuse_cache": false` exists for callers who need each call provably independent — a
determinism harness, a security boundary between tenants sharing a handle.

### The interaction that makes this dangerous

A cancelled generation leaves its partial output in the cache. The next request's prefix match
would extend a *truncated* assistant turn as though it were complete. This is silent: no error,
plausible output.

**Cancellation therefore rolls the cache back to the prompt boundary**, and the retained
sequence is truncated to match. The two features are one change for this reason.

### What is deliberately unsolved

Eviction. Every spike run fits in `n_ctx`. Reject / `n_keep` shift / evict is a real decision
with real trade-offs and it is not made here. Until it is, reuse stops helping at the window
boundary and behaviour degrades to today's — which is acceptable, and must be *stated*, not
discovered.

## Token counting

```c
const char* llb_count_tokens(llb_chat_t* chat, const char* request_json);
```

Takes the same `messages` (and optional `tools`) shape as infer, applies the chat template,
tokenizes, and returns `{"tokens": N, "n_ctx": M}`. No context decode.

**Why an entry point and not a binding helper:** it needs the model's vocab *and* its parsed
chat template. A binding has neither. That is the test for whether something belongs in the ABI.

**Why it takes a `chat` handle** rather than a path: `llb_model_info` already demonstrates the
load-and-free-again pattern, and paying a model load per token count would make the feature
useless for the budgeting loop it exists to serve.

## Cancellation

```c
typedef int (*llb_token_cb)(const char* token_piece, void* user_data);  /* non-zero => stop */
```

Returning non-zero ends generation with `"finish_reason": "cancelled"`. The response is still a
complete, well-formed response carrying the partial text and honest usage counts — a cancelled
generation is a result, not an error, and it consumed real tokens.

**Rejected:** a separate `llb_chat_cancel(handle)` called from another thread. It needs
cross-thread signalling the bridge does not otherwise have, and it cannot express "stop after
this token", which is what a stream consumer actually wants.

**Binding shape** — each language, its own idiom, same behaviour:

| binding | how a consumer cancels |
|---|---|
| Go | the callback returns `false`; a cancelled `context.Context` also stops |
| Python | the callback returns `False`, or raises |
| JS | the callback returns `false` |
| C# | the callback returns `false`; a signalled `CancellationToken` also stops |
| Java *(n/a)* | no Java binding — ADR-0006 |

Bindings that expose a native cancellation type (`context.Context`, `CancellationToken`) wire it
to the same return value. They add no behaviour the ABI lacks; they translate a native idiom
into the ABI's one mechanism.

## `llb_chat_create` config

```c
llb_chat_t* llb_chat_create(const char* gguf_path,
                            const char* config_json,   /* NULL => defaults */
                            llb_event_cb event_cb,
                            void* user_data);
```

Mirrors `llb_embed_create`, which already takes exactly this. `{"n_ctx":…, "n_batch":…,
"n_seq_max":…}`, all optional. `NULL` behaves as today.

`n_seq_max` has no observable effect in this change. It is here because it is a *create-time*
context parameter — the one thing that cannot be added later through request JSON — and the
whole reason slots can safely be deferred (ADR-0008 D6).

## Conformance

Beyond the per-binding suites, three assertions exist because the spike found the failure mode
first:

1. **Schema output is parsed.** Assert the returned content is valid JSON matching the schema —
   not that it "contains" the fields. The fence is invisible to a substring check.
2. **Prefix reuse does not change output.** Same conversation, `reuse_cache` true and false,
   same seed ⇒ identical text. Reuse is a latency feature; any output difference is a bug.
3. **Cancel-then-continue.** Cancel mid-generation, issue a different request, assert the answer
   is correct. This is the D2×D4 interaction, and it is the one bug in this change that is
   silent rather than loud.

Any benchmark calls `llama_synchronize`. `llama_decode` is async on Metal and CUDA; without it
the first spike run measured enqueue latency and reported prefix reuse as noise.

## Order of work

1. `llb_token_cb` + `llb_chat_create` signatures — the breaks, together, while they are free.
2. Prefix reuse **with** cancellation rollback — one unit, per the interaction above.
3. `json_schema` / `grammar` + parse-before-return.
4. `llb_count_tokens`.
5. Bindings, one capability at a time across all five, per ADR-0002 parity.
