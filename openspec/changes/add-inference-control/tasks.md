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

- [x] `llb_token_cb` returns `int`; non-zero stops generation. Header comment states the
      contract and that zero means continue.
- [x] `llb_chat_create` takes a `config_json` parameter (`n_ctx`, `n_batch`, `n_seq_max`), NULL
      ⇒ today's defaults. Mirror `llb_embed_create`'s shape exactly.
- [x] Bump `LLB_BRIDGE_VERSION`; both changes are breaking and must be visible through
      `llb_version`.
- [x] CHANGELOG `## Unreleased` — say plainly that these break, and what a caller does about it.

## 2. Prefix reuse + cancellation — ONE unit

They ship together because abort-without-rollback plus prefix reuse is a silent correctness
bug, not a slow path.

- [x] `llb_chat` retains the resident token sequence.
- [x] Longest-common-prefix match; `llama_memory_seq_rm(mem, seq, n, -1)`; decode only the tail.
- [x] Replace the unconditional `llama_memory_clear` at `llamabridge.cpp:339`.
- [x] `"reuse_cache": false` clears first — identical behaviour to a fresh engine.
- [x] Cancellation rolls the cache back to the prompt boundary and truncates the retained
      sequence to match.
- [x] `"finish_reason": "cancelled"` with the partial text and honest usage counts.
- [x] C-level test: same conversation with reuse on and off, same seed ⇒ identical text.
- [x] C-level test: cancel mid-generation, then a different request ⇒ correct answer.

## 3. Constrained output

- [x] `json_schema` on the request → `inputs.json_schema` → `COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT`.
- [x] `grammar` on the request → installed AFTER `common_chat_templates_apply` and typed
      `COMMON_GRAMMAR_TYPE_USER` (**not** prefilled). NOTE: setting `inputs.grammar` does
      NOT work on the jinja path — upstream assigns it (`common/chat.cpp:2354`), validates it
      (`:2413`), and then every format handler rebuilds `data.grammar` from scratch, so a
      caller's GBNF is silently discarded. We install it over `cparams.grammar` afterwards
      and clear the triggers/laziness that belonged to the grammar it replaced.
- [x] Stop hardcoding `COMMON_GRAMMAR_TYPE_TOOL_CALLS` at `llamabridge.cpp:381`; map by source.
- [x] Both fields present ⇒ a stable error code, no generation.
- [x] Parse before returning, so fenced-but-conformant output reaches the caller as JSON.
- [x] Test asserts the content **parses** and matches the schema — not that it contains the
      field names. A substring check cannot see the fence.

## 4. Token counting

- [x] `llb_count_tokens(chat, request_json)` → `{"tokens": N, "n_ctx": M}`.
- [x] No context decode, no cache mutation. Test asserts the cache is untouched.
- [x] Header documents the ownership rule like every other returned string.

## 5. Bindings — all five, per ADR-0002 parity

A capability lands everywhere or it is not done. Each binding is idiomatic, not a
transliteration.

- [x] **Go** — callback returns `bool` (*keep going*); `InferContext`/`InferStreamContext` for a
      cancellable `context.Context`; functional options `WithContextSize`/`WithBatchSize`/
      `WithMaxSequences`; `CountTokens`; `JSONSchema`/`Grammar`/`ReuseCache *bool`.
      `go vet` clean. **24 pass, 2 skip.**
- [x] **Python** — callback returns `False` (an `None` return CONTINUES); a raising callback
      stops and re-raises after the core's string is freed. **17 pass, 1 skip.**
- [x] **JS** — callback returns `false` (`undefined` continues); a SEPARATE koffi proto for the
      token callback, since reusing the void one would discard the return value silently.
      **19 pass, 1 skip**, exits cleanly with no forced teardown.
- [x] **C#** — `Func<string,bool>` callback and a `CancellationToken` that returns the partial
      response rather than throwing, because the core reports cancellation as a result with
      real usage counts. **21 pass, 1 skip.**
- [x] No Java binding (ADR-0006).
- [x] Cross-binding agreement test (`core/tests/agreement.sh`): Go, Python and JS run the same
      schema-constrained request at temperature 0 and must return byte-identical text AND an
      identical token count. **AGREE — 3 bindings.** Compares OUTPUT, not internals, so a
      binding that quietly reshapes a result fails here while passing its own suite.
      C# is not in the harness (it needs a compiled project rather than a script); its own
      suite covers the same assertions.
- [x] `task test` runs all of it — C ABI, four bindings, parity gate — with every model var
      wired. **102 checks, ZERO skips**: C 16, Go 26, Python 18, JS 20, C# 22, plus AGREE.
      The reranker and LoRA suites are no longer optional; `task models` reports what is
      missing rather than letting a suite skip quietly.

## 6. Docs

- [x] Header comments carry the contracts, including the two that are non-obvious: the callback
      return value, and that reuse is on by default.
- [x] `CHANGELOG.md` under `## Unreleased`, written as what a user gets.
- [x] Name what is NOT done and where it is tracked: slots and continuous batching (ADR-0008
      D6), cache eviction (task 0), session save/restore, logprobs, multimodal.

## 7. Not in this change — tracked, not dropped

- [ ] Slots + continuous batching. Additive via request JSON and the `n_seq_max` config key
      taken in task 1; measured at 2.1× for four slots. ADR-0008 D6. **Any implementation
      carries an explicit per-slot logits index and asserts output sanity, not distinctness.**
- [ ] Cache eviction policy (also task 0).
- [ ] Driving one context from multiple host threads. Untested; likely answer is "serialise at
      the bridge."
- [ ] Session save/restore, logprobs, multimodal — no consumer has asked.
