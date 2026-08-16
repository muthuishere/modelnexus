# Changelog

All notable changes to modelnexus are recorded here. This file is what a user reads to learn
what changed — it is written with the change, not reconstructed from the git log at release
time.

## Unreleased

### Breaking

Both of these change an exported C signature, so **every binding changes**. They are being taken
now, while there are effectively no external consumers, rather than after the documentation
tells people to depend on the current shape.

- **The streaming callback returns a value.** `llb_token_cb` was `void`; it now returns `int`,
  and returning non-zero stops generation. Until now a consumer that walked away — a cancelled
  context, a closed stream, a user pressing stop — could not tell the model, so it ran to
  completion and you paid for all of it. In each binding this is the language's own idiom:
  return `false` from the callback, and in Go and C# a cancelled `context.Context` /
  `CancellationToken` also stops.
- **`llb_chat_create` takes a configuration.** It accepts a new second parameter,
  `config_json`, carrying `n_ctx`, `n_batch` and `n_seq_max` — mirroring `llb_embed_create`,
  which already worked this way. Chat was previously not configurable at all. **Passing NULL is
  byte-identical to the old behaviour.**

### Added

- **Constrained output.** Give a request a `json_schema` and the model can only produce output
  matching it — the single most useful thing you can do with a small local model, and what makes
  one emit your exact shape instead of your shape plus an apology. A raw GBNF `grammar` is
  accepted too, for anything a JSON Schema cannot say. Supplying both is an error rather than a
  silent precedence rule. **The result is guaranteed to parse:** llama.cpp's generated grammar
  deliberately permits a markdown code fence, so the core strips it rather than handing you a
  "JSON" string that `json.loads` rejects.
- **Token counting.** `count_tokens` reports how large a message list is before you commit to a
  call, so context budgeting stops being guesswork. It applies the model's chat template and
  tokenizes — no inference, no context, ~6 ms for an 80-message conversation. It lives in the
  core because counting needs the model's vocabulary *and* its chat template, and no binding has
  either.
- **KV cache control.** Two calls — a status read and a clear — tell you how many tokens the
  engine is holding against its context window, and let you drop them. Cache reuse is right for a
  conversation that appends and wrong the moment a handle moves to unrelated work: the previous
  conversation keeps occupying context memory, and two tenants sharing a handle would share a
  cache. `reuse_cache: false` on the next request also clears, but only as a side effect of doing
  work, which is no help when you need the memory back now or need to prove the handle is empty
  before passing it on. The clear returns the state after it ran — always zero tokens — so you can
  assert the release happened instead of trusting it did. The status read changes nothing.
- **Cancelled generations return a result, not an error.** A cancelled call still gives you a
  complete response with the text produced so far, honest token counts, and
  `"finish_reason": "cancelled"`. You were charged for that work; you should get to see it.

### Changed

- **Conversations no longer re-read themselves every turn.** The engine keeps what is already in
  its KV cache and re-computes only the part of a new prompt that actually differs. For an agent
  loop — where each turn appends to an unchanging prefix — per-turn cost stops growing. Measured
  on a 1.5B model over 32 turns: **4198 ms of prefill became 829 ms, 9× by the last turn and
  still widening**, because reuse stays flat at ~26 ms per turn while the old path grew with the
  conversation. Output is unaffected; this is purely latency. Set `reuse_cache: false` when a
  call must be provably independent.

### Per binding

Each is the language's own idiom, not a transliteration — same behaviour, native shape.

| | stop a stream | cancel natively | create options |
|---|---|---|---|
| Go | callback returns `false` | `InferContext` with a `context.Context` | `WithContextSize` / `WithBatchSize` / `WithMaxSequences` |
| Python | callback returns `False`, or raises | — | `Chat(path, n_ctx=…, n_batch=…)` |
| JS | callback returns `false` | — | `new Chat(path, { nCtx, nBatch })` |
| C# | `Func<string,bool>` returns `false` | a signalled `CancellationToken` | `new Chat(path, nCtx: …, nBatch: …)` |

A callback that returns nothing (`None`, `undefined`) **continues** — only an explicit false
stops. In Python and C# a callback that throws stops generation and the exception is re-raised
after the native call unwinds, rather than crossing the C frame.

Cache control is two methods everywhere, named the way each language names things: Go
`CacheStatus` / `ClearCache`, Python `cache_status` / `clear_cache`, JS `cacheStatus` /
`clearCache`, C# `CacheStatus` / `ClearCache`. Both return the same pair — resident tokens and
the context window — because a token count only means something against the window it must fit
in.

**C# source-breaking**: the stream callback is `Func<string, bool>`; an `Action<string>` no
longer fits. No overload was kept, because the two would be ambiguous on an explicit `null` and
a void delegate hides the stop signal entirely.

### Not done

- **No slots and no continuous batching**, though both were measured and work — four concurrent
  conversations in one context are 2.1× faster batched into a single decode than run serially.
  They are additive later and need no further ABI break, which is why the `n_seq_max` config key
  above exists now. Tracked in ADR-0008 D6.
- **No cache-eviction policy.** What happens when a conversation outgrows the context window —
  reject, shift, or evict — is not decided. Until it is, cache reuse simply stops helping at the
  window boundary and behaviour falls back to the old path.
- **No session save/restore, token logprobs, or multimodal input.** Nobody has asked, and surface
  added speculatively is surface owned forever.

## 0.1.0 — 2026-08-15

First public release. Pre-alpha: the API may still move.

**Published: Go only.** `go get github.com/muthuishere/modelnexus/bindings/go@v0.1.0`, verified
from the public proxy with `Fetch()` pulling the natives and inference running. npm and PyPI
are built and verified locally but not published — they need registry credentials that only the
account owner can create; see `PUBLISHING.md`. Native platforms: linux-x86_64, darwin-aarch64,
windows-x86_64 (Intel macOS is staged by hand — GitHub's runners for it never start).

### Changed
- The PyPI **distribution** is `muthuishere-modelnexus`, not `modelnexus` — that name is held by an
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
- **Windows builds from source, not prebuilt binaries** — and it has to. llama.cpp's Windows
  release archive ships 29 DLLs and **zero `.lib` import libraries**, and MSVC cannot link a
  DLL without one, so CMake fails outright. Found the first time `build.cmd` ran in CI.
  `build.cmd` therefore defaults to source mode on Windows while `build.sh` still defaults to
  prebuilt on Unix. It is slow and confined to the rare natives workflow.
- **Intel macOS (`darwin-x86_64`) is not built in CI.** GitHub's Intel runners queued
  indefinitely — 50+ minutes with no start, repeatedly — and one job that never runs blocks the
  publish step for every platform that did. It is dropped from both matrices and staged by hand
  instead, which is what mochallama already does for the same platform; `PUBLISHING.md` has the
  command. Published platforms are linux-x86_64, darwin-aarch64 and windows-x86_64.
- **`natives.yml` now stages whatever built** (`if: always()`) instead of losing every platform
  because one failed or never got a runner. Intel macOS runners sit queued for a long time —
  mochallama seeds that platform by hand for the same reason — and `release.yml` already skips
  platforms that are not staged, so partial coverage degrades instead of blocking a release.
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
