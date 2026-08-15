# Spike 0001 — Is `llamabridge` portable as-is?

- **Date:** 2026-08-15
- **Verdict:** **Yes.** No Java-specific coupling in the ABI. Extraction is packaging work,
  not redesign.

## Question

mochallama's `llamabridge` was written for Java/Panama. Before committing to ADR-0002 (the C
ABI is the product) we needed to know: does the existing C surface assume a Java caller in a
way that would force a redesign before Go, Python, C# and JS could bind it?

Specifically:

1. Does anything JVM-specific cross the boundary (JNI types, JVM handles, Java object model)?
2. Is memory ownership expressible without a GC on the other side?
3. Are callbacks usable from FFIs with weaker function-pointer support than Panama?
4. How large is the surface — i.e. how much work is one binding?

## Method

Read the shipped surface in mochallama at
`core/src/main/cpp/include/llamabridge.h` (187 lines) and
`core/src/main/cpp/src/llamabridge.cpp` (703 lines), and inspected the declared contract.

## Findings

**1. No JVM coupling.** The header's own design goals state the intent: *"Pure C surface, no
C++ name-mangling, easy to bind from any FFI"* and *"Opaque handle pattern so the Java side
never sees llama.cpp types."* The implementation is C++ (it uses llama.cpp's `common_chat` /
`common_sampler`) but everything is exported `extern "C"`. No JNI type, no JVM handle, no Java
object crosses the boundary. The only Java-specific content is prose in the doc comments.

**2. Ownership is explicit and GC-free.** Strings cross as malloc'd UTF-8 and the caller
releases them through `llb_string_free`. Engines are opaque handles released by
`llb_chat_destroy`. Both are trivially expressible in Go, Python, C# and JS — this is the same
ownership pattern every FFI already handles.

**3. Callbacks are plain C function pointers** — `void (*)(const char*, void*)` — with a
`user_data` passthrough. Bindable from purego, ctypes, P/Invoke and koffi without exotic
support. This is the one area to verify empirically per language (streaming callbacks invoked
from a native thread interact with each runtime's threading model), which is spike 0002.

**4. The surface is small — seven functions and two callback typedefs:**

| symbol | role |
|---|---|
| `llb_chat_create` | load a GGUF, return an opaque engine handle (NULL on failure) |
| `llb_model_info` | pre-flight a GGUF's tool-calling capability without loading an engine |
| `llb_chat_infer` | single call: request JSON in, response JSON out |
| `llb_chat_infer_stream` | same, with a per-token callback |
| `llb_string_free` | release a returned string |
| `llb_chat_destroy` | release an engine |
| `llb_version` | static build identity (bridge + linked llama.cpp tag) |

Generation parameters, messages, tools, `tool_choice`, usage accounting and error reporting
all travel **inside the JSON**, not in the ABI. That is the single most important portability
property here: adding a parameter does not change the ABI, so bindings do not churn.

## What this means

- **ADR-0002 is safe to accept.** The bridge is already the thing we want to extract.
- **A binding is small.** Seven entry points, two of them frees. Days of work per language,
  not weeks — which is what makes five languages affordable for one maintainer.
- Extraction work is: move the sources, rename the symbol prefix if we choose to, build it
  standalone, and set up the native release. No redesign of the surface.

## Gaps this spike also exposed

The current surface is **chat-only**. The three capabilities that motivated modelnexus are all
absent from the ABI and must be added:

- **LoRA adapter** load / unload / scale
- **Embeddings**
- **Reranking**

Those are new entry points, and they are the real design work — not the extraction. Tracked in
`openspec/changes/extract-llamabridge-core/`.

## Not verified here

- Whether the *implementation* (`.cpp`, 703 lines) has hidden assumptions; only the declared
  contract was read.
- Threading behaviour of the streaming callback under each target runtime → **spike 0002**.
- Whether the standalone build cross-compiles cleanly outside Gradle → **spike 0003**.
