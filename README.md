# modelnexus

**Local LLM inference as a C ABI, bound into five languages.** llama.cpp underneath,
GGUF models, no daemon, no HTTP hop, no native-install dance.

> Status: **pre-alpha, nothing published — but it works.** The C core is extracted from
> [mochallama](https://github.com/deemwar-products/mochallama), where it has been running in
> production as `llamabridge` behind Java/Panama. Two bindings are working and tested against a
> real model; three are not written yet. See `docs/adr/` for the decisions and
> `openspec/changes/` for what is in flight.

## What this is

One thin `extern "C"` surface over llama.cpp — opaque handles, JSON in, JSON out, malloc'd
UTF-8 strings the caller frees — plus one thin binding per language:

| language | mechanism | status |
|---|---|---|
| Go | `purego`, no cgo | 14 tests, `CGO_ENABLED=0`, `go vet` clean |
| Python | `ctypes` | 8 tests |
| C# | P/Invoke, net8.0 | 14 tests |
| JS | `koffi` | 12 tests |

No Java binding, deliberately ([ADR-0006](docs/adr/0006-no-java-binding.md)) — the JVM is
served by [mochallama](https://github.com/deemwar-products/mochallama), which already ships a
Panama binding plus the Spring integration a JVM developer actually wants.

All four bindings run the same assertions and assert the **same error codes**. Verified
against a real 1.5B model and a real reranker: identical output, down to rerank scores of
`+6.591 / -5.161 / -11.004` in every language.

## What it does

| | |
|---|---|
| **chat** | generation, streaming, OpenAI-shaped tool calls, token usage |
| **LoRA** | load / scale / remove / clear adapters at runtime, several at once |
| **embeddings** | one vector per input, L2-normalized, choice of pooling |
| **reranking** | query-document scoring with a reranker model |
| **log control** | silence the engine, or route its output to your own logger |

### Getting the native library

Python and JS ship it inside the package — install and it works. Go resolves it at runtime,
because a Go module is a source tree and bundling five platforms would make every `go get`
pull ~70 MB nobody uses ([ADR-0007](docs/adr/0007-how-natives-reach-each-ecosystem.md)):

```go
dir, err := modelnexus.Fetch()   // downloads once into the user cache
```

or set `MODELNEXUS_LIB`, or build it yourself with `core/build.sh`. Skipping the step gives a
typed error listing every path searched — never a segfault.

```bash
task build     # download llama.cpp's prebuilt libs, compile only our bridge (seconds)
task verify    # confirm the ABI exports and every runtime dep resolves
task test      # run every binding's suite against a real model
```

Set `MODELNEXUS_MODEL` to a tool-capable GGUF to run the model-backed tests; without it
they skip rather than fail.

Inference runs **inside your process**. Nothing is served, nothing is spawned, nothing
leaves the box.

## What this is not

- **Not a model server.** If you want a shared GPU server with the widest catalogue, use
  Ollama. modelnexus is for embedding a model *inside* an application.
- **Not an agent framework.** It emits OpenAI-shaped `tool_calls`; it never executes them.
  That job belongs to [toolnexus](https://github.com/muthuishere/toolnexus), which does not
  depend on this library and never will (ADR-0003).
- **Not a multi-runtime abstraction.** llama.cpp only. ONNX will never be linked into this
  core (ADR-0001).

## The stack it belongs to

```
your app
├── toolnexus    → tools, MCP, agent skills, the calling loop   (7 languages)
├── modelnexus   → the local model, in-process                  (5 languages)
└── citenexus    → retrieval and evidence-gated answers
```

Three independent libraries. None imports another. They meet at the OpenAI wire shape.

## Layout

| path | what |
|---|---|
| `core/` | the C ABI — `include/modelnexus.h`, `src/`. The product. |
| `bindings/<lang>/` | one thin FFI binding per language. No logic. |
| `docs/adr/` | architecture decision records — **read these first** |
| `spikes/` | throwaway proofs that de-risk a decision before it becomes a spec |
| `openspec/` | spec-driven change workflow; behaviour is pinned here before it is coded |

## License

MIT
