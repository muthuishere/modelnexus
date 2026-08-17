# ADR 0009 — A model reference is a URI, and resolving it is the binding's job

- **Status:** Proposed (2026-08-17)
- **Date:** 2026-08-17
- **Driver:** `Open("model.gguf")` requires the file to already be on disk, which means every
  consumer writes the same download-and-cache code before they can write anything else.

## Context

Today the only thing that works is a local path:

```go
chat, _ := modelnexus.Open("qwen2.5-1.5b-instruct-q4_k_m.gguf")
```

Every real deployment then needs the step in front of it: fetch from Hugging Face on a laptop,
from S3 in production, from a mirror in an air-gapped build. That code is identical in every
consumer, gets it subtly wrong in different ways, and is the first thing anyone writes.

The obvious shape is a URI:

```
qwen2.5-1.5b-instruct-q4_k_m.gguf                       a local path, unchanged
hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-...gguf      Hugging Face
s3://bucket/models/qwen2.5-1.5b.gguf                    S3
https://internal.example/models/qwen.gguf               anything else
```

The question this ADR answers is **where the resolution happens**, because that decision is not
reversible once bindings ship it.

## Decision

**A model reference is a URI. The C ABI keeps taking a local path. Each binding resolves the
URI to a local file, then calls the core.**

The core gains nothing. `llb_chat_create` and `llb_embed_create` still take a path that exists.

### Why not in the core

Putting it in the core means putting **an HTTP client and an S3 client into a C++ library that
ships as a prebuilt binary for five platforms.** Concretely that is libcurl plus a TLS stack
plus an AWS SDK (or a hand-rolled SigV4), vendored, cross-compiled, and security-patched by us,
on every platform, forever. `LLAMA_CURL` is `OFF` in our build precisely because we did not want
the first of those.

The distribution cost is the argument, not taste. Our native closure is ~14 MB and has no
network dependency at all; that is why `natives-*` archives are auditable and why the licence
notices fit on one page. Adding a TLS stack changes the security surface of every consumer who
only wanted to run a model from disk.

Meanwhile every host language already has an excellent HTTP client and a first-party S3 SDK,
maintained by people who do that full time, with credential chains that already work in that
ecosystem.

### What this costs, and the rule that keeps it honest

This is **behaviour in a binding**, which ADR-0002 otherwise forbids. That is a real tension and
it is resolved the same way `§8`'s options were: the behaviour is **specified once and pinned by
the parity gate**, not left to each binding's judgement.

Specifically:

- **The URI grammar is normative.** `hf:`, `s3://`, `https://`, and "anything else is a path".
- **The cache layout is normative**, so two bindings on one machine share a download rather than
  each keeping their own copy of a 4 GB file.
- **The gate checks it.** Every binding resolves the same reference to the same cache path.

A binding that invents a scheme, or caches somewhere else, fails the gate. That is the
difference between "specified behaviour implemented per language" and "a binding doing whatever
it likes", and it is the whole reason this is allowed to live outside the core.

### Cache

```
<cache root>/models/<scheme>/<sha256 of the canonical URI>/<filename>
```

`MODELNEXUS_CACHE` overrides the root, as it already does for natives. A download is written to
a temporary name and **renamed into place**, so a half-downloaded 4 GB file can never be opened
as a model — the same rule the job bus and the parity gate already follow.

Resumable, and verified by size and ETag where the source provides them. A model that is already
present is not re-downloaded and not re-hashed.

### Credentials

**Never in the API.** S3 uses the ecosystem's default credential chain — environment,
`~/.aws/config`, instance role. Hugging Face uses `HF_TOKEN` from the environment for gated
repositories.

No modelnexus API takes a secret, ever. A signature that accepts a key is a signature that ends
up in someone's source control.

## Alternatives rejected

**Resolve in the core.** Rejected on distribution cost above.

**A separate `modelnexus-fetch` CLI.** Honest, and it does not help: the point is that
`Open("hf:…")` works in one line. A CLI is what people already write by hand.

**Only support Hugging Face.** Tempting — it is where GGUFs live. Rejected: S3 is what an
enterprise actually uses for a model it cannot put on a public hub, and that is the same user
who most needs an in-process runtime.

**Let `Open` take a URL and stream it.** Rejected. llama.cpp mmaps the file; a stream would mean
buffering the whole model in memory, which is exactly what mmap exists to avoid.

## Consequences

- Bindings gain a dependency on their language's HTTP and S3 SDKs. **Optional** where the
  ecosystem allows it (a Python extra, a Go build tag), so a consumer who only opens local files
  pays nothing.
- `Open` becomes a call that can take minutes and use the network on first use. It must report
  progress and be cancellable — a 4 GB silent hang is worse than the problem being solved.
- The first-use failure modes multiply: no network, bad credentials, gated repo, disk full,
  wrong quant. Each needs a typed error that says which, because "could not open model" is
  useless when the cause is an expired token.
- **Not decided here:** whether `hf:owner/repo` without a filename picks a quantisation
  automatically. Convenient and ambiguous; it deserves its own decision rather than being
  smuggled in.
