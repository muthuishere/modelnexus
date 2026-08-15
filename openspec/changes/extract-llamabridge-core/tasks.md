# Tasks

## 0. De-risk

- [ ] Spike 0002 — streaming callback across purego and Panama: is the token callback safe
      when invoked from a native thread? Record the verdict per runtime.
- [ ] Spike 0003 — does the standalone CMake build cross-compile for linux-x64, linux-arm64,
      darwin-arm64, darwin-x64, windows-x64 while consuming prebuilt llama.cpp libs?
- [ ] Read `llamabridge.cpp` (703 lines) end to end and record any assumption the header does
      not declare. Spike 0001 read only the contract.

## 1. Core

- [ ] Move `llamabridge.h` / `llamabridge.cpp` into `core/`, unchanged in substance.
- [ ] CMake build producing a shared library per platform, consuming prebuilt llama.cpp.
- [ ] Pin the llama.cpp tag explicitly and surface it through `llb_version`.
- [ ] A C smoke test that loads a small GGUF, infers once, and frees — no language runtime.

## 2. Java binding (Panama)

- [ ] Port mochallama's FFM binding into `bindings/java/`, behaviour-identical.
- [ ] Tests: create, model_info, infer, infer_stream, destroy, error JSON mapping.

## 3. Go binding (purego)

- [ ] `bindings/go/` — purego loader, `CGO_ENABLED=0` enforced in CI.
- [ ] Explicit typed error for "shared library not found / not initialized".
- [ ] Tests mirroring the Java set, asserting the **same error codes**.

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
