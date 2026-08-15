# modelnexus

**Local LLM inference as a C ABI, bound into five languages.** llama.cpp underneath,
GGUF models, no daemon, no HTTP hop, no native-install dance.

> Status: **pre-alpha, nothing published.** The C core is being extracted from
> [mochallama](https://github.com/deemwar-products/mochallama), where it has been running in
> production as `llamabridge` behind Java/Panama. See `docs/adr/` for the decisions and
> `openspec/changes/` for the work in flight.

## What this is

One thin `extern "C"` surface over llama.cpp — opaque handles, JSON in, JSON out, malloc'd
UTF-8 strings the caller frees — plus one thin binding per language:

| language | mechanism | status |
|---|---|---|
| Java | Panama FFM (JDK 22+) | proven in mochallama |
| Go | `purego` (no cgo) | planned |
| Python | `ctypes` | planned |
| C# | P/Invoke | planned |
| JS/TS | `koffi` / Node-API | planned |

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
