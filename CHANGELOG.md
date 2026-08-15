# Changelog

All notable changes to modelnexus are recorded here. This file is what a user reads to learn
what changed — it is written with the change, not reconstructed from the git log at release
time.

## Unreleased

### Added
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
- Staging copied `libllama.dylib` but not the **versioned** `libllama.0.dylib` /
  `libllama-common.0.dylib` the bridge actually links, producing a `dist/` that looked complete
  and failed to load. It now matches versioned names, and excludes llama.cpp's `*-impl`
  tool libraries, which are not runtime dependencies.

### Not done yet
- **No Java, C#, or JS binding.** Java is proven in mochallama but has not moved here.
- **No LoRA, embeddings, or reranking.** The ABI is chat-only — these are the three
  capabilities that motivated this repo and none of them exists yet. Tracked in
  `openspec/changes/extract-llamabridge-core/`.
- **No log control.** llama.cpp writes verbosely to stderr and the ABI has no way to quiet it,
  so an embedding application inherits engine chatter it cannot switch off. Needs an ABI
  addition; recorded in spike 0002.
- **Nothing is published**, and no release workflow exists. mochallama still vendors its own
  copy of the bridge and is unchanged.
- The header does not document that `llb_chat_create` **retains** the event callback. That is
  an ABI documentation defect, found the hard way in spike 0002, and should be fixed in the
  core rather than worked around per binding.
- Windows is unbuilt: `core/build.sh` covers macOS and Linux only.
