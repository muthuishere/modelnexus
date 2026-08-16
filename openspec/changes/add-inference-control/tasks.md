# Tasks

## 0. De-risk — done

- [x] Spike 0003 — all five questions measured against prebuilt llama.cpp b9371.
      Verdicts: schema YES (nearly free — upstream already has the fields), prefix reuse YES
      (9.0× at turn 32, flat ~26 ms vs linear growth), token counting YES (5.9 ms), cancel YES
      (with cache rollback), slots YES (2.1× batched over serial, deferred as additive).
- [x] Three traps recorded before they could ship: the schema grammar permits a markdown fence;
      the grammar *type* decides prefill and the bridge hardcodes one; per-slot sampling with
      `idx = -1` reads the wrong slot's logits and produces distinct-looking garbage.
- [x] Methodology: `llama_decode` is async on Metal/CUDA — measurements must synchronize.
- [ ] Decide the cache-eviction policy for a conversation that outgrows `n_ctx` (reject /
      `n_keep` shift / evict). **Not blocking this change** — reuse degrades to today's
      behaviour at the boundary — but it must be decided before a long session meets it.

## 1. ABI signature changes — take both together, while they are free

- [ ] `llb_token_cb` returns `int`; non-zero stops generation. Header comment states the
      contract and that zero means continue.
- [ ] `llb_chat_create` takes a `config_json` parameter (`n_ctx`, `n_batch`, `n_seq_max`), NULL
      ⇒ today's defaults. Mirror `llb_embed_create`'s shape exactly.
- [ ] Bump `LLB_BRIDGE_VERSION`; both changes are breaking and must be visible through
      `llb_version`.
- [ ] CHANGELOG `## Unreleased` — say plainly that these break, and what a caller does about it.

## 2. Prefix reuse + cancellation — ONE unit

They ship together because abort-without-rollback plus prefix reuse is a silent correctness
bug, not a slow path.

- [ ] `llb_chat` retains the resident token sequence.
- [ ] Longest-common-prefix match; `llama_memory_seq_rm(mem, seq, n, -1)`; decode only the tail.
- [ ] Replace the unconditional `llama_memory_clear` at `llamabridge.cpp:339`.
- [ ] `"reuse_cache": false` clears first — identical behaviour to a fresh engine.
- [ ] Cancellation rolls the cache back to the prompt boundary and truncates the retained
      sequence to match.
- [ ] `"finish_reason": "cancelled"` with the partial text and honest usage counts.
- [ ] C-level test: same conversation with reuse on and off, same seed ⇒ identical text.
- [ ] C-level test: cancel mid-generation, then a different request ⇒ correct answer.

## 3. Constrained output

- [ ] `json_schema` on the request → `inputs.json_schema` → `COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT`.
- [ ] `grammar` on the request → `inputs.grammar` → `COMMON_GRAMMAR_TYPE_USER` (**not**
      prefilled).
- [ ] Stop hardcoding `COMMON_GRAMMAR_TYPE_TOOL_CALLS` at `llamabridge.cpp:381`; map by source.
- [ ] Both fields present ⇒ a stable error code, no generation.
- [ ] Parse before returning, so fenced-but-conformant output reaches the caller as JSON.
- [ ] Test asserts the content **parses** and matches the schema — not that it contains the
      field names. A substring check cannot see the fence.

## 4. Token counting

- [ ] `llb_count_tokens(chat, request_json)` → `{"tokens": N, "n_ctx": M}`.
- [ ] No context decode, no cache mutation. Test asserts the cache is untouched.
- [ ] Header documents the ownership rule like every other returned string.

## 5. Bindings — all five, per ADR-0002 parity

A capability lands everywhere or it is not done. Each binding is idiomatic, not a
transliteration.

- [ ] **Go** — callback returns `bool`; a cancelled `context.Context` also stops. `ChatOptions`
      for the create config. `CountTokens`. `JSONSchema` / `Grammar` on the request.
- [ ] **Python** — callback returns `False`, or raises. Same four capabilities.
- [ ] **JS** — callback returns `false`. Same four.
- [ ] **C#** — callback returns `false`; a signalled `CancellationToken` also stops. Same four.
- [ ] No Java binding (ADR-0006).
- [ ] Cross-binding agreement test: identical schema-constrained output and identical token
      counts across all four, the way the rerank scores already agree byte-for-byte.

## 6. Docs

- [ ] Header comments carry the contracts, including the two that are non-obvious: the callback
      return value, and that reuse is on by default.
- [ ] `CHANGELOG.md` under `## Unreleased`, written as what a user gets.
- [ ] Name what is NOT done and where it is tracked: slots and continuous batching (ADR-0008
      D6), cache eviction (task 0), session save/restore, logprobs, multimodal.

## 7. Not in this change — tracked, not dropped

- [ ] Slots + continuous batching. Additive via request JSON and the `n_seq_max` config key
      taken in task 1; measured at 2.1× for four slots. ADR-0008 D6. **Any implementation
      carries an explicit per-slot logits index and asserts output sanity, not distinctness.**
- [ ] Cache eviction policy (also task 0).
- [ ] Driving one context from multiple host threads. Untested; likely answer is "serialise at
      the bridge."
- [ ] Session save/restore, logprobs, multimodal — no consumer has asked.
