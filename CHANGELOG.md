# Changelog

All notable changes to modelnexus are recorded here. This file is what a user reads to learn
what changed — it is written with the change, not reconstructed from the git log at release
time.

## Unreleased

### Added
- **LoRA adapters at runtime** — load, rescale, remove and list adapters on a live engine,
  several at once, without reloading the model. One ABI entry point with a JSON op dispatch,
  so adding an operation later does not churn four bindings. Adapters change *behaviour* —
  output format, tone, tool-call reliability — not knowledge; for facts, retrieve.
- **Embeddings** — one vector per input, in input order, L2-normalized by default so a dot
  product is a cosine similarity. Pooling is selectable (`mean` / `cls` / `last` / `none`).
- **Reranking** — query-document scoring with a reranker model. Results come back sorted
  best-first, each carrying the document's *original* index so you can map back to your own
  list. Requires `pooling: "rank"`, and refuses to run without it rather than returning
  numbers that look like scores and are not.
- **C# binding** (P/Invoke, net8.0) and **JavaScript binding** (koffi) — joining Go and
  Python. All four run the same assertions and assert the same error codes.
- **The native core builds and runs.** `core/` holds the `extern "C"` bridge over llama.cpp,
  extracted from mochallama unchanged in substance, with `core/build.sh` replacing Gradle as
  the orchestrator so a Go or Python consumer needs no JVM toolchain. Default mode downloads
  llama.cpp's official prebuilt release libraries and compiles only the bridge — seconds, not
  an engine build. `--source` builds llama.cpp too, for when you want the whole thing yourself.
- **Python binding** (`ctypes`) — `Chat`, `model_info`, `version`, streaming via `on_token`,
  context-manager lifecycle, typed `ModelError` carrying the core's stable error code.
- **Go binding** (`purego`, no cgo) — `Open`, `Infer`, `InferStream`, `Info`, `Version`, typed
  `Error`, and a distinct `ErrNativeLibraryNotFound` listing every path searched. Builds and
  vets clean with `CGO_ENABLED=0`, so cross-compilation still works.
- **Taskfile** — `task build`, `verify`, `test`, `dist`, `clean`, plus `llama:pinned` /
  `llama:latest` / `llama:pin TAG=…` for moving the pinned llama.cpp release deliberately.
  Nothing auto-updates: a silent engine bump would change inference output underneath users.
- ADR-0001 … ADR-0005, and spikes 0001 (the bridge is portable as-is) and 0002 (callback
  lifetime — the core retains the event callback, and a binding that forgets crashes).

### Fixed
- `task test` passed the model path with a literal `~`, which no runtime except Python
  expands. The Go suite silently **skipped** every model-backed test while reporting
  success, and the C# run aborted. Paths are now resolved to absolute through the shell,
  and `task models` reports up front whether the test models are present — a suite that
  skips everything should not look identical to one that passes.
- Staging copied `libllama.dylib` but not the **versioned** `libllama.0.dylib` /
  `libllama-common.0.dylib` the bridge actually links, producing a `dist/` that looked complete
  and failed to load. It now matches versioned names, and excludes llama.cpp's `*-impl`
  tool libraries, which are not runtime dependencies.

### Not done yet
- **No Java binding, deliberately** — ADR-0006. The JVM is served by mochallama, which
  already ships a Panama binding and the Spring integration a JVM developer wants.
- **LoRA is untested against a real adapter.** The operations, the error paths and the
  apply/rollback logic are exercised, but no `.gguf` adapter has been loaded — only the
  failure paths. Producing one is `modelforge tune`'s job, which does not exist yet.
- **Embedding runs one sequence per decode.** Correct, and slower than batching many
  sequences into one call. A later optimisation, not a correctness gap.
- **The C# binding has a smoke run, not a test suite** like the other three.
- **No log control.** llama.cpp writes verbosely to stderr and the ABI has no way to quiet it,
  so an embedding application inherits engine chatter it cannot switch off. Needs an ABI
  addition; recorded in spike 0002.
- **Nothing is published**, and no release workflow exists. mochallama still vendors its own
  copy of the bridge and is unchanged.
- The header does not document that `llb_chat_create` **retains** the event callback. That is
  an ABI documentation defect, found the hard way in spike 0002, and should be fixed in the
  core rather than worked around per binding.
- Windows is unbuilt: `core/build.sh` covers macOS and Linux only.
