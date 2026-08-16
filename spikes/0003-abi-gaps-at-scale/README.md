# Spike 0003 — the four ABI gaps, and whether the design scales

## The questions

The ABI is 14 entry points and chat-shaped. Four things a serving stack has and we
do not, plus one question about whether our *shape* can ever grow into one:

| | question |
|---|---|
| **Q1** | Can a caller-supplied JSON schema constrain output, and what does it cost us? |
| **Q2** | Does KV prefix reuse actually pay across a long agent conversation? `llamabridge.cpp:339` clears the cache on every call. |
| **Q3** | Can we count tokens for a message list cheaply, without decoding? |
| **Q4** | Can generation be stopped mid-flight? `llb_token_cb` returns `void` (`llamabridge.h:54`), so today nothing can. |
| **Q5** | Can one context hold N independent conversations — the shape every serving stack has and a one-handle-one-conversation ABI does not? |

## Method

One throwaway C++ program (`spike.cpp`) linking the **same** prebuilt llama.cpp b9371 the
bridge links. No bridge code involved: the point is what the substrate can do, not what our
wrapper currently does.

```
./run.sh            # all five, default qwen2.5-1.5b-instruct-q4_k_m
./run.sh prefix     # one question
```

Measured on an M-series Mac, Metal + Accelerate, `n_ctx=8192`. Raw output: `results.txt`.

**One methodology note that changed the answer.** `llama_decode` is **asynchronous** on
Metal and CUDA. The first run of Q2 timed enqueue latency rather than work and came back
as noise (`reuse_ms` bouncing 8→75 ms for a fixed 48-token tail), which read as "reuse does
not pay." Adding `llama_synchronize` after each decode turned it into a flat line. Any
benchmark we publish must synchronize, or it measures nothing.

---

## Verdicts

### Q1 — YES, and it is nearly free. **Verdict: adopt.**

`common_chat_templates_inputs` already has `json_schema` and `grammar` fields
(`common/chat.h:191-192`). Upstream does the schema→GBNF conversion for us. The bridge
simply never sets them.

```
baseline      Paris is the capital city of France. It has a population of
              approximately 2.2 million people. Paris was founded in 514 AD.

CONSTRAINED   ```json
              { "city": "Paris", "population": 2, "founded": 256 }
              ```
              parses + matches schema exactly: YES
```

**Two traps, both found here rather than in production:**

1. **The generated grammar permits a markdown fence.** Its root rule is
   ``root ::= "<|im_start|>assistant\n" space space ("```json" space response-format space "```" | response-format)``.
   Fenced output is *grammar-conformant*, so a core that returns raw text hands the caller a
   string that is not JSON. The core must parse before returning.
2. **The grammar type is not cosmetic.** `llamabridge.cpp:381` hardcodes
   `COMMON_GRAMMAR_TYPE_TOOL_CALLS`. A schema grammar is `OUTPUT_FORMAT` and a raw
   caller-supplied GBNF is `USER`; they differ in whether the generation prompt is prefilled
   into the grammar sampler (`common_grammar_needs_prefill`, `common/common.h:204`). Passing a
   user GBNF as `TOOL_CALLS` prefills a grammar that must not be prefilled.

### Q2 — YES, decisively, and it gets better the longer the conversation. **Verdict: adopt.**

24 turns of a growing agent conversation, prompt prefix never changing — exactly the shape a
tool-calling loop produces.

| turn | prompt tokens | clear (today) | reuse | reused | speed-up |
|---|---|---|---|---|---|
| 1 | 45 | 25.1 ms | 24.1 ms | 0 | 1.0× |
| 8 | 374 | 54.7 ms | 25.1 ms | 327 | 2.2× |
| 16 | 757 | 121.5 ms | 26.1 ms | 709 | 4.7× |
| 24 | 1141 | 190.2 ms | 26.6 ms | 1093 | 7.2× |
| 32 | 1525 | 247.2 ms | 27.3 ms | 1477 | **9.0×** |

**The shape matters more than the ratio.** Clearing costs grow linearly with prompt length —
so total prefill over a conversation is quadratic. Reuse is **flat at ~26 ms** regardless of
how long the conversation has run. Over 32 turns: 4198 ms → 829 ms, and the gap was still
widening when the run ended. On a 1.5B model at 1525 tokens. Both numbers get worse with a
bigger model and a longer context.

Mechanism is `llama_memory_seq_rm(mem, seq, n_common_prefix, -1)` then decoding only the
divergent tail. No new llama.cpp capability required.

### Q3 — YES, and it costs ~6 ms. **Verdict: adopt.**

81 messages → 2084 tokens in 5.92 ms (4.56 template apply + 1.36 tokenize). No context, no
decode. It needs the model's vocab **and** its chat template, which is exactly why a host
cannot do this itself and why it belongs in the ABI rather than in each binding.

### Q4 — YES, and it is a signature change, not a mechanism. **Verdict: adopt, and do it before 1.0.**

Aborting is just breaking the sampling loop; there is no llama.cpp support required. What the
spike checked is the part that could have bitten: **is the cache still usable afterwards?**

```
aborted after 12 tokens; cache holds 54 positions (prompt 42 + 12 generated)
rolled back to position 42, ran a different request, reused 24 of 38 prompt tokens
next request after abort: "Paris"
```

Yes — provided the abort is followed by rolling the cache back to the prompt boundary. This
interacts directly with Q2: **abort without rollback leaves a partial generation in the cache
that a later prefix match would happily reuse.** The two features must land together or Q2
becomes a correctness bug.

`llb_token_cb` must become `int (*)(const char*, void*)` — non-zero means stop. That is an ABI
break, which is free now and expensive after the docs site tells people to depend on it.

### Q5 — YES. One context, N conversations, and batching is where the throughput is.

This is the "does it scale like the serving stacks" question. It does, and llama.cpp gives us
the machinery today — `n_seq_max` on the context, per-sequence memory ops, and per-row logits.

```
4 slots × 24 tokens:  serial 642 ms (150 tok/s)   batched 305 ms (314 tok/s)   2.1×
```

Same work; the only difference is whether the four slots' tokens go into one `llama_decode` or
four. **2.1× at four slots on a laptop**, and the gap widens with slot count because a decode
is dominated by weight movement, not by batch width.

**The bug this spike caught is the important part.** The first version sampled with `idx = -1`
("last logits") after decoding each slot separately — so slot 0 read slot 3's logits. Output:

```
slot 0: The221 greater.
slot 1:  number1ndrst than
```

Four distinct streams of confident garbage. A naive `distinct != identical` check calls that a
pass. **Any per-slot design must carry an explicit logits index**, and its conformance test must
assert the content is sane, not merely that the slots differ.

---

## What this means for the ABI

Q1–Q4 are all confirmed and all cheap. Nothing here needs a llama.cpp change, a second runtime,
or a redesign — the substrate already does everything; the bridge just does not expose it.

Q5 is the one with a real architectural consequence: **`llb_chat_t` is one handle = one
conversation = one sequence**, and that is a dead end for anything server-shaped. The spike says
the fix is not a rewrite — it is a sequence id where today there is an implicit `0`. Adding the
concept while there is one caller is a footnote; adding it after five bindings ship is a
breaking change in five languages at once.

Ordering follows from the evidence, not from taste:

1. **Q4 first** — it is a signature change, and it is the cheapest thing to do before people depend on the ABI.
2. **Q2 with it** — because abort-without-rollback plus prefix-reuse is a correctness bug, not a slow path.
3. **Q1** — highest user-visible value, small surface.
4. **Q3** — one entry point.
5. **Q5** — decide the *shape* now (does a handle carry a sequence id?) even if slots ship later.

## Not answered here

- **Cache eviction.** Every run fits in `n_ctx`. What happens when a conversation outgrows the
  window — reject, shift with `n_keep`, or evict — is untested and is a design decision, not a
  measurement.
- **Threading.** All measurements are single-threaded. Whether slots can be driven from N host
  threads against one context is a separate question, and the answer is probably "no, serialise
  at the bridge."
- **Non-Metal backends.** CUDA is also async so the synchronize finding should hold, but it was
  not measured.
- **Bigger models.** Every number is from a 1.5B. The Q2 and Q5 gaps should widen; that is an
  argument from mechanism, not a measurement.
