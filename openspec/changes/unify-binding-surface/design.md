# Design — one rule per binding, and a gate that can see it

## The principle underneath all four

A binding marshals, calls, and frees. Every one of these defects is a binding making a
decision that belongs to the core, or declining to make one that belongs to it:

| defect | who decided what |
|---|---|
| Python forwards unknown kwargs | the binding declined to validate, so the core silently absorbed it |
| JS translates two keys | the binding decided which names are its own and which are the wire's |
| `n_batch: 512` sent unasked | the binding restated a default the core owns |
| the gate ignores creation | the gate only checks the half where disagreement is visible |

## 1. Python rejects unknown keyword arguments

```python
def infer(self, messages, *, max_tokens=None, temperature=None, ..., **params):
    ...
    request.update(params)      # today
```

`**params` exists so a caller can reach a core parameter the binding has not named yet — which
is a real requirement, because ADR-0002's whole point is that a new generation parameter costs
no binding change. Removing it would break that.

**Decision: keep the escape hatch, make it explicit.** Unknown *keyword* arguments raise
`TypeError`. A caller who genuinely wants an unnamed core parameter passes a dict:

```python
chat.infer(messages, max_tokens=64)                    # named — checked
chat.infer(messages, extra={"mirostat": 2})            # deliberate — passed through
chat.infer(messages, max_token=64)                     # TypeError: unknown parameter
```

The distinction is intent. `max_token=64` is a typo that looks like a parameter; `extra={...}`
cannot be typed by accident.

**Rejected:** validating against a list of known core parameters. The binding would then need
updating whenever the core gains one, which is the coupling the JSON surface exists to avoid.

## 2. JS is camelCase, in and out

One rule: **the JS API is camelCase; the wire is snake_case; the boundary translates
everything.** No key is exempt.

```js
// in
{ messages, maxTokens: 64, topK: 40, jsonSchema, reuseCache: false }
// out
{ type, text, toolCalls, finishReason, usage: { promptTokens, completionTokens, totalTokens } }
```

Implementation is a general recursive `snake_case ↔ camelCase` converter at the boundary rather
than a per-key rename table, because a rename table is how you end up with two conventions
again: the table is what has to be remembered, and it will be forgotten for the next key.

**One exception, stated rather than discovered:** a caller-supplied `jsonSchema` is **passed
through untouched**. It is not our data — it is a JSON Schema whose property names are the
caller's, and converting them would rewrite the contract the caller is asking the model to
satisfy. Property order matters under grammar-constrained decoding (see the Go note below), so
does spelling.

**Rejected:** accepting both spellings during a deprecation window. Nothing is published on npm
yet, so there is nobody to deprecate for — and accepting both is how a codebase ends up with
both forever.

## 3. Only what the caller set

Every binding builds its create config from explicitly supplied values only. `Embedder()` with
no arguments sends `NULL`, not `{}`, and certainly not `{"n_batch":512}`.

This makes the core the single home for every default, which is already the rule for
`llb_chat_create` (its header says NULL must behave exactly as before the parameter existed).
`llb_embed_create` gets the same treatment.

The cost is real and worth naming: a binding can no longer document "the default is 512" from
its own source. It documents "the model's default" and points at the ABI. That is correct —
a default restated in four places is four things that can drift, and this change exists because
one of them already had.

## 4. The parity gate learns about creation

`core/tests/agreement.sh` compares inference text and token counts. It gains a second phase:
each binding prints the **create config it would send** for three intents — no options, a
context size only, and a full config — and the payloads must match exactly.

That is a stronger check than comparing behaviour, because two bindings sending different
configs can still behave identically today and diverge on the next core change. The gate should
fail when they *disagree*, not when the disagreement finally becomes visible.

It also gains an observable check: create with an explicit `n_ctx` and assert every binding
reports the same `n_ctx` back through `count_tokens`. That is the only externally visible proof
the config crossed the ABI at all — a binding that marshalled the config and dropped it would
otherwise still load, still infer, and still look correct.

## A note the Go binding should carry

Not a defect in this change, but found alongside them and worth writing down where a Go user
will see it: `encoding/json` **sorts map keys**, so a JSON Schema passed as `map[string]any`
reaches the model with its properties reordered. Under grammar-constrained decoding property
order is load-bearing — the model commits to fields in the order the grammar allows — so the
same prompt and seed produced a different value in Go than in Python and JS. The fix at the
call site is `json.RawMessage`. `Request.JSONSchema` is `any` and a map is legal, so this is a
documentation duty, not an API change.

## Order of work

1. The gate first. A gate written after the fixes only proves the fixes; a gate written before
   proves the defect was real.
2. Python kwargs, JS casing, create-config omission — independent of each other.
3. Re-run the gate.
