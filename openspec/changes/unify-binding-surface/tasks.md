# Tasks

## 0. The gate first

A gate written after the fixes only proves the fixes. A gate written before proves the defect
was real.

- [x] Extend `core/tests/agreement.sh` with a **create-config phase**: each binding prints the
      config it would send for three intents — no options, context size only, full config —
      and the payloads must match exactly across bindings.
- [x] Add an **observable** check: create with an explicit `n_ctx`, assert every binding reports
      the same `n_ctx` back through `count_tokens`. A binding that marshals config and drops it
      would otherwise still load, still infer, and still look correct.
- [x] **Confirm the gate FAILS on today's code** before fixing anything. It did, and it found
      THREE divergences, not the one reported: `go-cfg chat-none={}` against `null` everywhere
      else, and `embed-none={"n_batch":512}` in Python and JS against `{}` in Go.
- [x] Add **C#** to the create-config phase (`Smoke --create-config`). It was the one binding
      the gate could not see, and that is exactly where an `Embedder` unconditionally sending
      `{"n_batch":512}` survived unnoticed — the same defect as Python's, in the binding nobody
      was checking.
- [x] A probe that FAILS now fails the gate. Dropping it silently turned a broken binding into
      an absent one, and the summary still said AGREE about the survivors.

## 1. Python — a typo is not a parameter

- [x] `Chat.infer()` / `infer_stream()` raise `TypeError` on an unrecognised keyword argument.
- [x] Keep an explicit pass-through for unnamed core parameters — a dict, not `**kwargs`, so it
      cannot be reached by a typo.
- [x] Same treatment anywhere else the binding does `request.update(...)`.
- [x] Test: `max_token=64` (singular) raises and sends nothing; `extra={"mirostat":2}` is sent
      unchanged.
- [x] Docstrings say which parameters are named and how to reach one that is not.

## 2. JS — one convention, in and out

- [x] A general recursive snake_case ↔ camelCase converter at the boundary (`src/wire.js`).
      **Not** a per-key rename table — the table is the thing that gets forgotten for the next key.
- [x] Every request option camelCase: `maxTokens`, `topK`, `topP`, `minP`, `repeatPenalty`,
      `jsonSchema`, `reuseCache`, `toolChoice`.
- [x] Every response field camelCase: `finishReason`, `toolCalls`,
      `usage.{promptTokens,completionTokens,totalTokens}`, `nCtx`.
- [x] **`jsonSchema` is passed through untouched.** Its property names are the caller's and
      describe the output contract; renaming them rewrites what the model was asked for.
      Property order is load-bearing under grammar-constrained decoding.
      EXTENDED beyond the brief: `tools[].function.parameters` is the same case — a
      caller-authored schema — and a blind converter would have rewritten every tool's
      parameter names, silently, and only in JS.
- [x] Test: a snake_case request option is **rejected**, with a message naming the camelCase
      spelling. Chosen because every core parameter is reachable in camelCase — including ones
      the binding has never named (`mirostatTau` -> `mirostat_tau`) — so no request exists that
      can only be written in snake_case, which makes a snake_case key a mistake and nothing else.
- [x] Update `bindings/js/README` and every JS sample in `site/` and `examples/`.

## 3. Every binding — send only what was asked for

- [x] Python `Embedder` stops sending `n_batch: 512` unasked; no options ⇒ NULL config.
- [x] JS `Embedder` likewise.
- [x] Go: send NULL, not `{}`, when the caller supplied nothing — they are not the same path
      through the core.
- [x] C#: was NOT already correct. `Chat` built its config properly; `Embedder` hardcoded
      `nBatch = 512` and always sent it. Two rules in one file.
- [x] `Chat` create config in all four: only explicitly supplied keys.
- [x] Each binding's docs say "the model's default", not a restated number, and point at the ABI.

## 4. Go — the schema-ordering trap

Not a defect in this change, but found alongside them and invisible until two languages are
compared.

- [x] Doc comment on `Request.JSONSchema`: `encoding/json` **sorts map keys**, so a schema
      passed as `map[string]any` reaches the model reordered. Property order is load-bearing
      under grammar-constrained decoding — the same prompt and seed produced a different answer
      in Go than in Python and JS. Use `json.RawMessage` to preserve order.
- [x] Same note on the structured-output docs page.

## 5. Re-verify

- [x] The gate passes.
- [x] `task test` — zero failures, zero skips, across C, Go, Python, JS, C#.
- [x] `task examples` — 24/24, and the JS examples updated for the new casing.
- [x] `cd site && npm run build` green, with every JS sample corrected.
- [x] `CHANGELOG.md` under `## Unreleased`: the JS and Python changes are **breaking** and must
      say so plainly, with what a caller does about it.

## 6. Gate defects found while using it

- [x] Fixed temp probe filenames inside the repo meant two concurrent runs deleted each other's
      probe through the EXIT trap, and a correct binding reported MODULE_NOT_FOUND. Now
      PID-suffixed.
- [x] A probe that FAILED was silently dropped, so a broken binding became an absent one and the
      summary still said AGREE about the survivors. It now fails the gate.
- [x] The summary said "AGREE — 3 bindings" after C# joined the config phase, which read as
      though C# had not participated. It now reports both counts.

## 7. Not in this change

- [ ] Whether `Embedder` should take the same functional-options shape as `Chat` in Go. A
      consistency question, not a correctness one.
- [ ] Whether the agreement gate should also cover embeddings and rerank output. It covers chat
      only today; the rerank scores are already known to agree byte-for-byte but nothing
      enforces it.
