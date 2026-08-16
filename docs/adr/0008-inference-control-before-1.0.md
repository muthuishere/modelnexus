# ADR 0008 — Close the four inference-control gaps before 1.0, and take the two signature breaks now

- **Status:** Accepted (2026-08-16)
- **Date:** 2026-08-16
- **Evidence:** [spike 0003](../../spikes/0003-abi-gaps-at-scale/) — five questions measured
  against prebuilt llama.cpp b9371, the same library the bridge links. Raw output committed as
  `results.txt`; `./run.sh` reproduces it.
- **Supersedes nothing.** Extends the ABI defined in ADR-0002.

## Context

The shipped ABI is 14 entry points and chat-shaped. Comparing it against what the reference
bindings expose ([llama-cpp-python](https://github.com/abetlen/llama-cpp-python),
[node-llama-cpp](https://node-llama-cpp.withcat.ai/)) found four things missing, and one
question about whether our *shape* can grow into a serving stack at all. Spike 0003 measured
all five. Every verdict came back positive and nothing requires a llama.cpp change: **the
substrate already does all of it; the bridge does not expose it.**

Two of the four cannot be done additively, which is why this is a decision and not a backlog
item. The ABI is the product (ADR-0002) and a breaking change breaks applications. Right now
there are five bindings and effectively no external consumers. That window closes when the
docs site ships.

## Decision

### D1 — Accept a caller-supplied JSON schema, and a raw GBNF grammar

`common_chat_templates_inputs` already carries `json_schema` and `grammar`
(`common/chat.h:191-192`); the bridge never sets them. Both become optional fields on the
infer request JSON. No exported symbol changes.

**Two consequences the spike forced into the open:**

1. **The core parses before it returns.** The grammar upstream generates *permits* a markdown
   fence — its root rule is
   ``root ::= "<|im_start|>assistant\n" space space ("```json" space response-format space "```" | response-format)``.
   Fenced output is therefore grammar-conformant, and a core returning raw text hands the
   caller a string that is not JSON. Observed, not theorised.
2. **The grammar *type* is load-bearing.** `llamabridge.cpp:381` hardcodes
   `COMMON_GRAMMAR_TYPE_TOOL_CALLS`. A schema grammar is `OUTPUT_FORMAT`; a caller-supplied
   GBNF is `USER`. They differ in whether the generation prompt is prefilled into the grammar
   sampler (`common_grammar_needs_prefill`, `common/common.h:204`). We map each source to its
   own type rather than reusing the one we already had.

### D2 — Reuse the KV cache across calls by longest common prefix

`llamabridge.cpp:339` clears the whole cache on every call, so a conversation re-prefills
itself from scratch every turn. Per-turn cost is linear in prompt length, which makes total
prefill over a conversation **quadratic**.

Measured over 32 turns of a growing agent conversation:

| turn | prompt | clear (today) | reuse | speed-up |
|---|---|---|---|---|
| 8 | 374 tok | 54.7 ms | 25.1 ms | 2.2× |
| 16 | 757 tok | 121.5 ms | 26.1 ms | 4.7× |
| 24 | 1141 tok | 190.2 ms | 26.6 ms | 7.2× |
| 32 | 1525 tok | 247.2 ms | 27.3 ms | **9.0×** |

Total 4198 ms → 829 ms, **and still widening when the run ended**. The ratio is not the
point; the shape is. Reuse is *flat* at ~26 ms regardless of conversation length. On a 1.5B
model at 1525 tokens — the gap grows with model size and context length.

Mechanism: `llama_memory_seq_rm(mem, seq, n_common_prefix, -1)`, then decode only the
divergent tail. Reuse is the **default**; a request field disables it for callers who need
each call provably independent.

### D3 — Add a token-counting entry point

81 messages → 2084 tokens in 5.9 ms, no context and no decode. It needs the model's vocab
**and** its chat template, so no binding and no host can compute it — which is exactly the
test for whether something belongs in the ABI. One new entry point.

### D4 — `llb_token_cb` returns `int`; non-zero stops generation. **Breaking.**

Today the callback returns `void` (`llamabridge.h:54`), so a consumer that walks away — a
cancelled context, a closed stream, a user pressing stop — cannot stop the model. It runs to
completion and bills for it.

Aborting needs no llama.cpp support; it is breaking the sampling loop. What the spike
verified is the part that could have bitten: **the cache is still usable afterwards, provided
the abort rolls it back to the prompt boundary.**

**This couples to D2 and the two must land together.** An abort that leaves a partial
generation in the cache creates a prefix a later request will happily match. D2 without D4's
rollback is not a slow path, it is a correctness bug.

### D5 — `llb_chat_create` takes a config JSON. **Breaking.**

`llb_chat_create(gguf_path, event_cb, user_data)` accepts no configuration at all — `n_ctx`,
`n_batch` and `n_seq_max` are not settable. `llb_embed_create` already takes a config JSON;
this makes chat consistent with it, and is the prerequisite for D6.

### D6 — Slots are deferred, and that deferral is safe **because** of ADR-0002

Spike 0003 confirmed one context holds N independent conversations, and that batching them
into a single decode is where throughput lives: **4 slots × 24 tokens, serial 642 ms
(150 tok/s) → batched 305 ms (314 tok/s), 2.1×**, widening with slot count.

I previously argued this had to be settled now or it became "a breaking change in five
languages." That was wrong, and the correction changes the plan: **because parameters travel
in JSON (ADR-0002), a `session` field on the infer request and an `n_seq_max` key in D5's
config are both additive.** Slots can land later without an ABI break. What we take now is
D5 — the config parameter that makes it possible — and nothing else.

What the spike says we must *not* do is ship a per-slot design that samples with "last
logits". Sampling `idx = -1` after per-slot decodes makes slot 0 read slot N-1's logits, and
the observed output was four *distinct* streams of confident garbage — `"The221 greater."`,
`" number1ndrst than"` — which a distinctness assertion passes. **Any future slot work carries
an explicit per-slot logits index, and its conformance test asserts sanity, not difference.**

### D7 — Benchmarks synchronize

`llama_decode` is asynchronous on Metal and CUDA. The first Q2 run measured enqueue latency,
came back as noise, and read as "prefix reuse does not pay." Any published measurement calls
`llama_synchronize`, or it measures nothing.

## Alternatives rejected

**Ship the docs site first, break the ABI at 0.2.0.** Rejected. D4 and D5 are signature
changes in five bindings. They cost nothing today and cost every consumer later; the docs site
is precisely the event that creates those consumers.

**Do D1 only — the marketing headline.** Rejected. D2 is the one that decides whether a
20-turn agent session is usable, and it is invisible in a demo, which is exactly why it gets
skipped and then never done.

**Add session save/restore, logprobs, multimodal in the same pass.** Rejected. None was
required by a consumer, each carries its own format and lifetime questions, and the ABI is the
product — surface added speculatively is surface owned forever.

**Build slots now.** Rejected per D6: additive later, and the cost of getting it wrong
(silent per-slot logits corruption) argues for doing it deliberately rather than alongside four
other changes.

## Consequences

- **Two ABI signature changes**, both before any external consumer exists: `llb_token_cb`
  gains an `int` return, `llb_chat_create` gains a config parameter. Every binding changes.
- **One new entry point** (token counting). Entry points go 14 → 15.
- **Everything else is request-JSON**, so it costs no signature anywhere.
- **Behaviour changes by default**: the KV cache is reused rather than cleared. Output is
  unchanged; only latency and cache state differ. Callers needing provable independence set
  the opt-out.
- **Parity rule applies** (ADR-0002): this lands in the ABI and all five bindings, or it is not
  done.
- **Deferred, tracked, not silently dropped**: slots and continuous batching (D6), cache
  eviction when a conversation outgrows `n_ctx`, driving one context from multiple host
  threads.

## Not answered by the evidence

- **Cache eviction.** Every spike run fits in `n_ctx`. Reject / shift with `n_keep` / evict is
  a design decision, not a measurement, and it must be made before D2 meets a long session.
- **Threading.** All measurements single-threaded. Whether one context can be driven from N
  host threads is untested; the likely answer is "no — serialise at the bridge."
- **Non-Metal backends.** CUDA is also async so D7 should hold, but it was not measured.
- **Bigger models.** Every number is from a 1.5B. D2 and D6 should improve with size; that is
  an argument from mechanism, not a measurement.
