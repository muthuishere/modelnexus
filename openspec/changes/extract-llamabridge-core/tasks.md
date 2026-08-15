# Tasks

## 0. De-risk

- [x] Spike 0002 — callbacks across ctypes and purego. Verdict: both work; the core RETAINS
      the event callback for the engine's life, so a local trampoline segfaults. Panama,
      P/Invoke and koffi still unverified.
- [ ] Spike 0003 — does the standalone CMake build cross-compile for linux-x64, linux-arm64,
      darwin-arm64, darwin-x64, windows-x64 while consuming prebuilt llama.cpp libs?
- [ ] Read `llamabridge.cpp` (703 lines) end to end and record any assumption the header does
      not declare. Spike 0001 read only the contract.

## 1. Core

- [x] Move `llamabridge.h` / `llamabridge.cpp` into `core/`, unchanged in substance.
- [x] CMake build producing a shared library per platform, consuming prebuilt llama.cpp
      (`core/build.sh`; macOS + Linux — Windows still to do).
- [x] Pin the llama.cpp tag explicitly and surface it through `llb_version` (b9371).
- [ ] A C smoke test that loads a small GGUF, infers once, and frees — no language runtime.

## 1b. Python binding (ctypes) — added, was not in the original plan

- [x] `bindings/python/` — loader with an explicit `NativeLibraryNotFound`, `Chat`,
      `model_info`, `version`, streaming, typed `ModelError`.
- [x] 7 tests green against a real GGUF; they skip, not fail, without `MODELNEXUS_MODEL`.

## 2. Java binding (Panama)

- [ ] Port mochallama's FFM binding into `bindings/java/`, behaviour-identical.
- [ ] Tests: create, model_info, infer, infer_stream, destroy, error JSON mapping.

## 3. Go binding (purego)

- [x] `bindings/go/` — purego loader, builds and vets clean with `CGO_ENABLED=0`.
- [x] Explicit typed error for "shared library not found / not initialized"
      (`ErrNativeLibraryNotFound`, listing every path searched).
- [x] Tests mirroring the Python set, asserting the **same error codes**. 7 pass.

## 4. Parity

- [ ] A shared fixture set both bindings run against, producing identical results — the same
      discipline as toolnexus's `examples/`.
- [ ] Document any specified divergence (e.g. Go's runtime-not-found error) in the spec, not
      in a code comment.

## 5. Native release (ADR-0004)

- [ ] Tier-1 workflow: build the native closure per platform into a durable, tag-keyed
      prerelease. Triggered only by `core/**`.
- [ ] Seed darwin-x64 locally via `gh` (no CI runner exists).
- [ ] Confirm tier 2 is *not* wired in this change — publishing is out of scope.

## 6. Docs

- [ ] `README.md` status section reflects what actually works.
- [ ] `CHANGELOG.md` entry under `## Unreleased`, naming what is not done.
- [ ] Per-binding README with a runnable example.

## Explicitly out of scope

- [ ] ~~LoRA / embeddings / rerank ABI~~ → separate changes
- [ ] ~~Python / C# / JS bindings~~ → separate change
- [ ] ~~Switching mochallama to consume the core~~ → separate change (ADR-0005)
- [ ] ~~Registry publishing~~ → separate change

## Found during implementation (new work)

- [ ] The header does not document that `llb_chat_create` **retains** the event callback for
      the engine's lifetime. Fix the doc comment in the core — every binding will otherwise
      rediscover this as a crash (spike 0002).
- [ ] No log control in the ABI. llama.cpp writes verbosely to stderr and an embedding host
      cannot quiet it. Needs an `llb_set_log_*` entry point.
- [ ] Windows support in `core/build.sh` (macOS and Linux only today).
