# Extract llamabridge into a standalone C core with two bindings

## Why

Local LLM inference is currently reachable from exactly one language in this portfolio —
Java — not because the code is Java-specific but because it lives inside a Spring product.
Spike 0001 established that mochallama's `llamabridge` C surface has no JVM coupling: opaque
handles, JSON in, JSON out, malloc'd UTF-8 with an explicit free, seven entry points. It was
written to be bound from any FFI and it is.

Meanwhile the capabilities that make a local model useful in an application — LoRA adapters,
embeddings, reranking — are absent from the ABI entirely, and cannot be added sensibly while
the core's home is a Gradle subproject of a Spring library.

This change moves the core out and proves it with a second, non-JVM binding. Nothing else is
worth building until that is true, because every later capability multiplies across bindings.

## What changes

- A standalone C core at `core/`, built without Gradle, with a stable symbol prefix.
- The **Java binding** (Panama) moved into this repo, behaviour-identical to mochallama's.
- A **Go binding** via `purego` — no cgo — as the proof that the surface is genuinely portable.
- A native build producing per-platform prebuilt libraries, staged per ADR-0004.
- mochallama is **not** switched over in this change. It keeps its vendored copy until the core
  is proven; the switch is a separate change so a regression there cannot be caused by this one.

## What this change does NOT do

- **No LoRA, embeddings, or reranking.** The ABI stays chat-only. Those are the point of the
  repo but they are separate changes, each landing in every binding — adding them mid-extraction
  would mean debugging new behaviour and new packaging at once.
- **No Python, C#, or JS binding.** Two bindings prove portability; five is release work.
- **No publishing.** Nothing goes to a registry in this change.
- **No change to mochallama.**

## Impact

- New: `core/`, `bindings/java/`, `bindings/go/`, native build workflow.
- Unchanged: mochallama, toolnexus, citenexus.
- Risk concentrated in the streaming callback across the FFI boundary (spike 0002) and in
  cross-compiling the standalone build (spike 0003). Both are spiked before implementation.
