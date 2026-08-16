# Examples

Eight examples, in Go, Python and JavaScript — the three published targets. The same
eight in each language, in the same order, so two languages read side by side.

These are files that run. `examples/run.sh` executes every one of them and CI runs it,
so an example that stops working fails the build rather than rotting in a doc.

| example | what it shows |
|---|---|
| `hello` | load a GGUF, one inference, print the answer |
| `streaming` | tokens as they arrive, and stopping early from the callback — a cancelled run returns a complete response with `finish_reason: cancelled` |
| `structured` | a JSON Schema in, output that is guaranteed to parse out, decoded into a real object |
| `conversation` | eight turns, per-turn wall clock, run twice — with KV prefix reuse and without |
| `counting` | count a message list against `n_ctx` before inferring, and what one tool declaration costs |
| `embeddings` | embed sentences and compare them; vectors are L2-normalised, so a dot product is the cosine |
| `rerank` | score documents against a query with a reranker model, best first, each keeping its original index |
| `lora` | load an adapter onto a live engine, infer, clear it, infer again |

## Running them

Point the environment at real GGUF files. An example whose model is missing is skipped
loudly and counted in the summary — it is never silently green.

```bash
export MODELNEXUS_MODEL=/path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf   # tool-capable chat model
export MODELNEXUS_RERANKER=/path/to/bge-reranker-v2-m3-Q4_K_M.gguf   # reranker, "rank" pooling
export MODELNEXUS_LORA_BASE=/path/to/qwen2.5-3b-instruct-q4_k_m.gguf # the base the adapter was built for
export MODELNEXUS_LORA=/path/to/adapter.gguf                         # the adapter itself

task build          # the native bridge, if you have not already
task examples       # all three languages
task examples:go    # or one: examples:go / examples:python / examples:js
```

`task examples` passes the repo's existing `MODEL` / `RERANKER` / `LORA` / `LORA_BASE`
variables through, so with the models in their default location the exports above are
unnecessary.

One at a time:

```bash
cd examples/go     && go run ./hello
cd examples/python && PYTHONPATH=../../bindings/python python3 hello.py
cd examples/js     && npm install && node hello.js
```

The LoRA example needs a **matched pair** — an adapter is built for one architecture and
will not load against an arbitrary model — which is why it takes two variables.

## How these resolve the library

The Go module and the JS package point at the bindings in this tree (a `replace`
directive and a `file:` dependency), so a change to a binding breaks the examples in the
same commit. Outside the repo those two lines become `go get` and a version range.
`run.sh` also exports `MODELNEXUS_LIB` at `core/dist/<platform>`, so the examples load
the library this tree built rather than one left in a package cache.
