# Give callers control over inference: schema, cache, counting, cancellation

## Why

The ABI can start a generation and watch it happen. It cannot **shape** it, **stop** it, or
**measure** it, and it throws away the work it did on the last call.

Spike 0003 measured four gaps against the same prebuilt llama.cpp b9371 the bridge links, and
every verdict came back positive — none requires a llama.cpp change or a redesign. The
substrate already does all of it; the bridge does not expose it:

- **No caller-supplied JSON schema.** A grammar is synthesised only from the chat template's
  tool-call path. `common_chat_templates_inputs` already has `json_schema` and `grammar` fields
  (`common/chat.h:191-192`) and we never set them.
- **The KV cache is cleared on every call** (`llamabridge.cpp:339`). A conversation re-prefills
  itself from scratch every turn — linear per turn, quadratic over a session. Measured:
  **4198 ms → 829 ms over 32 turns, 9.0× at turn 32 and still widening**, with reuse flat at
  ~26 ms regardless of how long the conversation has run.
- **No token counting.** Nothing can ask how large a message list is, so every context-budgeting
  decision in every consumer is a guess. It costs 5.9 ms and needs the model's vocab *and* chat
  template, so no binding can do it itself.
- **Generation cannot be stopped.** `llb_token_cb` returns `void` (`llamabridge.h:54`). A
  consumer that walks away — cancelled context, closed stream, user pressed stop — cannot tell
  the model, which runs to completion and bills for it.

Two of these are signature changes. Right now there are five bindings and effectively no
external consumers; that window closes when the docs site ships. See
[ADR-0008](../../../docs/adr/0008-inference-control-before-1.0.md).

## What changes

- **Request JSON gains `json_schema` and `grammar`.** The core maps each to its correct
  `common_grammar_type` and **parses the result before returning it**, because the generated
  grammar permits a markdown fence and conformant output is therefore not always parseable JSON.
- **KV prefix reuse becomes the default.** Longest common prefix against the cached sequence,
  `llama_memory_seq_rm` the divergent tail, decode only what changed. A request field opts out.
- **A new entry point counts tokens** for a message list without creating a context or decoding.
- **`llb_token_cb` returns `int`** — non-zero stops generation. **Breaking.** Stopping also rolls
  the cache back to the prompt boundary, which is why it must land with prefix reuse rather than
  after it.
- **`llb_chat_create` takes a config JSON** (`n_ctx`, `n_batch`, `n_seq_max`), matching
  `llb_embed_create`. **Breaking.** Today chat is not configurable at all.
- All of it lands in the ABI **and all five bindings**, per ADR-0002.

## What this change does NOT do

- **No slots and no continuous batching.** Spike 0003 confirmed both work — 4 slots batched into
  one decode is 2.1× over serial — but because parameters travel in JSON they are **additive
  later** and need no ABI break. This change takes only the `n_seq_max` config key that makes
  them possible. Tracked in ADR-0008 D6.
- **No cache eviction policy.** What happens when a conversation outgrows `n_ctx` — reject,
  shift with `n_keep`, or evict — is a design decision this change does not make. It must be
  made before prefix reuse meets a genuinely long session; until then reuse degrades to today's
  behaviour at the window boundary.
- **No session save/restore, no logprobs, no multimodal.** No consumer asked; each carries its
  own format and lifetime questions; surface added speculatively is surface owned forever.
- **No multi-threading of one context.** Untested, and the likely answer is "serialise at the
  bridge."

## Impact

- **Breaking**: `llb_token_cb` signature, `llb_chat_create` signature. Every binding changes.
- **New**: one entry point (token counting). 14 → 15.
- **Behaviour**: the KV cache is now reused rather than cleared. Output is unchanged; latency
  and cache state differ. Opt out per request when independence must be provable.
- **Unchanged**: toolnexus, mochallama, the embed/rerank surface, the release machinery.
- Risk is concentrated in prefix reuse meeting cancellation — the one interaction where a bug
  is silent rather than loud, and the reason those two ship together.
