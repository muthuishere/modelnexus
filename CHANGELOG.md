# Changelog

All notable changes to modelnexus are recorded here. This file is what a user reads to learn
what changed — it is written with the change, not reconstructed from the git log at release
time.

## Unreleased

### Changed
- The PyPI **distribution** is `modelnexus-core`, not `modelnexus` — that name is held by an
  unrelated project ("Global AI Model Vault", 1.0.4) and PyPI normalises case-insensitively.
  The **import is unchanged**: `import modelnexus`. npm (`@muthuishere/modelnexus`) and the Go
  module path are unaffected.

### Added
- **Log control** — `set_log_level` / `SetLogLevel` and a handler hook in every binding.
  llama.cpp writes hundreds of lines to stderr per model load; the bridge now owns that sink,
  **defaults to WARN rather than the engine's default**, and lets a host silence it entirely or
  route it into their own logger. A library embedded in someone else's process should be quiet
  unless asked.
- **Batched embedding** — several sequences per decode instead of one, chunked to respect
  `n_batch` and `n_seq_max`. Vectors are bit-identical to the unbatched path; measured 1.2x on
  a 1.5B model, more on small dedicated embedders where per-decode overhead dominates.
- **Windows build** — `core/build.cmd`, the batch counterpart to `build.sh`, driving the same
  CMake with the same two modes. Plain `cmd` rather than PowerShell: execution policy never
  blocks a `.cmd`, and Windows 10 1803+ ships `curl` and `tar` (bsdtar reads zip), so nothing
  needs installing.
- **Publishing** — three GitHub workflows: `ci.yml` (build + all four suites on every push,
  including an ABI completeness check), `natives.yml` (tier 1: compile the per-platform native
  closure into a durable, tag-keyed prerelease), `release.yml` (tier 2: **never compiles**;
  downloads the staged natives and publishes to Go, PyPI and npm). Each leg is gated on a repo
  variable, so check the job list — a green run with skipped legs is not a release.
- **`modelnexus.Fetch()`** in Go — downloads the platform's natives into the user cache once.
- **A C# test suite** (14 xunit tests) replacing the smoke run.
- **LoRA verified against a real adapter** — load, rescale, stack two, remove, infer with one
  applied, clear. Previously only the failure paths were exercised.
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
- Staging dereferenced llama.cpp's versioned symlinks, turning one 7.5 MB library into three
  copies. `dist/` was 40 MB per platform; preserving the links makes it **14 MB**, which is
  what makes bundling into wheels and npm packages practical at all.
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
- **Nothing is published yet.** The workflows exist and are valid, but no release has been
  cut and the `ENABLE_GO` / `ENABLE_PYPI` / `ENABLE_NPM` repo variables are unset, so every
  publishing leg would skip.
- **Windows is written but unrun.** `core/build.cmd` and the CI matrix entry exist; no Windows
  machine has executed either, here or in CI.
- **mochallama has not been switched** to consume this core. It keeps its own copy of the
  bridge and ships unchanged — deliberately staged, per ADR-0005.
- **No NuGet publishing.** The C# binding works and is tested; it is simply not published.
- **No log control.** llama.cpp writes verbosely to stderr and the ABI has no way to quiet it,
  so an embedding application inherits engine chatter it cannot switch off. Needs an ABI
  addition; recorded in spike 0002.
- **Nothing is published**, and no release workflow exists. mochallama still vendors its own
  copy of the bridge and is unchanged.
- The header does not document that `llb_chat_create` **retains** the event callback. That is
  an ABI documentation defect, found the hard way in spike 0002, and should be fixed in the
  core rather than worked around per binding.
- Windows is unbuilt: `core/build.sh` covers macOS and Linux only.
