# Tasks

## 0. The gate first

A gate written after the fixes only proves the fixes. A gate written before proves the defect
was real.

- [ ] Extend `core/tests/agreement.sh` with a **create-config phase**: each binding prints the
      config it would send for three intents — no options, context size only, full config —
      and the payloads must match exactly across bindings.
- [ ] Add an **observable** check: create with an explicit `n_ctx`, assert every binding reports
      the same `n_ctx` back through `count_tokens`. A binding that marshals config and drops it
      would otherwise still load, still infer, and still look correct.
- [ ] **Confirm the gate FAILS on today's code** before fixing anything. Record what it says.

## 1. Python — a typo is not a parameter

- [ ] `Chat.infer()` / `infer_stream()` raise `TypeError` on an unrecognised keyword argument.
- [ ] Keep an explicit pass-through for unnamed core parameters — a dict, not `**kwargs`, so it
      cannot be reached by a typo.
- [ ] Same treatment anywhere else the binding does `request.update(...)`.
- [ ] Test: `max_token=64` (singular) raises and sends nothing; `extra={"mirostat":2}` is sent
      unchanged.
- [ ] Docstrings say which parameters are named and how to reach one that is not.

## 2. JS — one convention, in and out

- [ ] A general recursive snake_case ↔ camelCase converter at the boundary. **Not** a per-key
      rename table — the table is the thing that gets forgotten for the next key.
- [ ] Every request option camelCase: `maxTokens`, `topK`, `topP`, `minP`, `repeatPenalty`,
      `jsonSchema`, `reuseCache`, `toolChoice`.
- [ ] Every response field camelCase: `finishReason`, `toolCalls`,
      `usage.{promptTokens,completionTokens,totalTokens}`, `nCtx`.
- [ ] **`jsonSchema` is passed through untouched.** Its property names are the caller's and
      describe the output contract; renaming them rewrites what the model was asked for.
      Property order is load-bearing under grammar-constrained decoding.
- [ ] Test: a snake_case request option is rejected or has no effect — pick one, state which,
      and pin it. Silently ignoring it is what we are removing.
- [ ] Update `bindings/js/README` and every JS sample in `site/` and `examples/`.

## 3. Every binding — send only what was asked for

- [ ] Python `Embedder` stops sending `n_batch: 512` unasked; no options ⇒ NULL config.
- [ ] JS `Embedder` likewise.
- [ ] Go and C#: confirm they already omit; add a test rather than assuming.
- [ ] `Chat` create config in all four: only explicitly supplied keys.
- [ ] Each binding's docs say "the model's default", not a restated number, and point at the ABI.

## 4. Go — the schema-ordering trap

Not a defect in this change, but found alongside them and invisible until two languages are
compared.

- [ ] Doc comment on `Request.JSONSchema`: `encoding/json` **sorts map keys**, so a schema
      passed as `map[string]any` reaches the model reordered. Property order is load-bearing
      under grammar-constrained decoding — the same prompt and seed produced a different answer
      in Go than in Python and JS. Use `json.RawMessage` to preserve order.
- [ ] Same note on the structured-output docs page.

## 5. Re-verify

- [ ] The gate passes.
- [ ] `task test` — zero failures, zero skips, across C, Go, Python, JS, C#.
- [ ] `task examples` — 24/24, and the JS examples updated for the new casing.
- [ ] `cd site && npm run build` green, with every JS sample corrected.
- [ ] `CHANGELOG.md` under `## Unreleased`: the JS and Python changes are **breaking** and must
      say so plainly, with what a caller does about it.

## 6. Not in this change

- [ ] Whether `Embedder` should take the same functional-options shape as `Chat` in Go. A
      consistency question, not a correctness one.
- [ ] Whether the agreement gate should also cover embeddings and rerank output. It covers chat
      only today; the rerank scores are already known to agree byte-for-byte but nothing
      enforces it.
