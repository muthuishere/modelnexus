# Make the bindings agree where they currently only look like they agree

## Why

Writing the documentation and the examples for 0.2.0 surfaced four inconsistencies. None is a
crash. Every one of them is the kind of defect that a per-binding test suite cannot see,
because each binding is individually correct — they only disagree with **each other**.

That is precisely the class of bug this project exists to prevent, and the reason the parity
gate exists. The gate did not catch these because of the fourth one.

1. **A typo is free in Python.** `Chat.infer()` forwards unknown keyword arguments straight
   into the wire request (`request.update(params)`). `max_token=16` — singular — is
   marshalled, sent, ignored by the core, and the call succeeds with the default 256. Nothing
   reports anything. Go will not compile that mistake and C# will not either; Python and the
   core between them make it silent.

2. **JS asks for two spellings in one object.** `toWire()` renames exactly `jsonSchema` and
   `reuseCache`; every other option passes through raw. So a JS caller writes
   `{ jsonSchema, max_tokens, top_k }` — camelCase next to snake_case, in the same literal,
   with no rule to remember it by. Responses are inconsistent in the other direction:
   `finish_reason` and `usage.completion_tokens` come back snake, but `countTokens()`
   hand-converts `n_ctx` to `nCtx`. One method translating and no other is a mistake a reader
   makes once per file. The agent writing the docs made it, in its own example, and had to fix
   it — which is the evidence.

3. **"No options" means three different payloads.** Creating an `Embedder` with no arguments
   sends `{"n_batch":512}` from Python, `{}` from Go, and `{"n_batch":512}` from JS. Same core,
   same intent, three different requests. Harmless today because the core's default happens to
   be 512 — and it stays harmless right up until the core changes its default, at which point
   two bindings silently pin the old value and one follows.

4. **The parity gate does not cover creation.** `core/tests/agreement.sh` compares inference
   output and token counts. It has nothing to say about the config a binding sends at create
   time, which is exactly why (3) survived to be found by a human reading three files side by
   side.

## What changes

- **Python rejects unknown keyword arguments** instead of forwarding them. A parameter the
  binding does not know is a caller error, and saying so costs one `if`.
- **JS is camelCase on the way in and on the way out.** Every request option is camelCase and
  translated at the boundary; every response field is camelCase. One rule, no exceptions,
  including `finishReason` and `usage.completionTokens`.
- **Every binding omits what the caller did not set.** Create config carries only explicitly
  supplied keys. The core owns every default, and a default has exactly one home.
- **The parity gate covers creation**, not just inference: the bindings must produce the same
  create-config payload for the same intent, and the same observable engine.

## What this change does NOT do

- **No change to the core's defaults**, only to who is allowed to state them.
- **No change to the C ABI.** Nothing here needs a new symbol; this is entirely about what the
  bindings do with the surface that already exists.
- **No renaming in Go, Python or C#.** Their spellings are already idiomatic and internally
  consistent. Only JS has two conventions in one object.

## Impact

- **Breaking for JS callers** who pass snake_case request options or read snake_case response
  fields. This is the whole of the fix and cannot be done compatibly: accepting both spellings
  would preserve the confusion permanently.
- **Breaking for Python callers** who currently pass a misspelled parameter and have not
  noticed, which is the point.
- **Behaviour-neutral for Go and C#.**
- All of it lands before 0.2.0 ships, while there are effectively no external consumers.
