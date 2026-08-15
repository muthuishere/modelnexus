# ADR-0001 — One core, one runtime: llama.cpp only

- **Status:** accepted
- **Date:** 2026-08-15

## Context

modelnexus exists to serve three capabilities to five languages: text generation, LoRA
adapters, and embeddings/reranking for retrieval. Two runtimes could plausibly provide them —
llama.cpp (GGUF) and ONNX Runtime.

The original sketch was to host ONNX and reach it from every language. Examination of the
existing portfolio changed the answer:

- Everything already built is llama.cpp/GGUF — mochallama, `local-tool-llm`, `ark`.
- llama.cpp has **native LoRA adapter support** with runtime apply/scale through a stable C
  API, reachable from every target language. ONNX Runtime does not: hot-swappable adapters
  live in `onnxruntime-genai`, a *different* library, whose bindings are C/C++, Python, C# and
  Java — **no Go binding**. Go is the primary language for this work, so the feature we most
  want would be unavailable where we most need it.
- llama.cpp also covers embeddings and reranking, so the third capability needs no second
  engine either.

## Decision

**The core links llama.cpp and nothing else.** ONNX Runtime will never be linked into this
core, in any build mode, behind any flag.

If in-process ONNX is ever genuinely required, it will be a **separate core** reusing this
repo's distribution machinery — not a mode, a backend, or a compile-time option of this one.

ONNX keeps one home in the portfolio: `modelforge`'s browser export target, where GGUF has no
story and ONNX Runtime Web does. That is a build-time conversion concern, not a runtime one.

## Consequences

- The native closure stays one library per platform. A second engine would mean two prebuilt
  natives × 5 platforms × 5 bindings — twenty artifacts instead of ten, on every release.
- The ABI can expose llama.cpp's full feature set. An abstraction over two engines could only
  expose their *intersection*, which excludes LoRA — the capability that motivated the repo.
- We inherit llama.cpp's release cadence and must pin a tag deliberately, as mochallama does
  (`b9371`).
- We give up ONNX's hardware-backend breadth (DirectML, CoreML EP, TensorRT). Accepted:
  llama.cpp's Metal/CUDA/Vulkan coverage matches our CPU-first, small-tool-model target.

## Alternatives rejected

- **Dual-runtime core with a backend abstraction.** The "unified inference abstraction" trap:
  years of work to expose the intersection of two feature sets, with double the native
  surface and no capability we cannot already get.
- **ONNX-only core.** Discards every existing asset, and loses LoRA in Go entirely.

## Open verification

The claim that `onnxruntime-genai` still has no Go binding, and llama.cpp's exact adapter and
rerank API surface at our pinned tag, are stated from knowledge and **have not been verified
against current releases**. Neither changes the decision — llama.cpp wins on portfolio fit
alone — but confirm before quoting either in user-facing material.
