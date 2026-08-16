# Third-party notices

modelnexus itself is MIT (see `LICENSE`). This file records everything else, split by
whether we **redistribute** it — because that is what decides whether a notice has to
ship inside our artifacts.

## Redistributed in our binaries — notices ship with every package

These are inside `libllamabridge` or sit beside it in `core/dist/<platform>/`, so their
notices are copied into that directory and travel into every wheel, npm package and the
`natives-*` release archives.

| component | licence | how it reaches us | notice |
|---|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (incl. ggml) | MIT — © 2023-2026 The ggml authors | the bridge links `libllama` / `libllama-common`, and the ggml backends ship alongside | `core/licenses/LICENSE-llama.cpp` |
| [nlohmann/json](https://github.com/nlohmann/json) | MIT — © 2013-2025 Niels Lohmann | **header-only, compiled into the bridge** via `#include "nlohmann/json.hpp"` (vendored inside llama.cpp) | `core/licenses/LICENSE-nlohmann-json` |

nlohmann/json is easy to miss precisely because it is header-only: nothing links against
it at runtime, but its code is in our compiled library, and MIT requires the notice "in
all copies or substantial portions of the Software."

llama.cpp vendors other libraries — cpp-httplib, miniaudio, stb, sheredom — but the
bridge includes none of them, and the CLI/server DLLs that would use them are explicitly
excluded from `dist/` (`! -name "*-impl*"`). They are not redistributed.

## Dependencies resolved by the user's package manager — not redistributed

We declare these; the user's toolchain fetches them from their own registry. Their
notices are not ours to bundle, and are listed here for transparency.

| binding | dependency | licence |
|---|---|---|
| Go | [ebitengine/purego](https://github.com/ebitengine/purego) | Apache-2.0 |
| JS | [koffi](https://github.com/Koromix/koffi) | MIT |
| C# | xunit, Xunit.SkippableFact *(test-only)* | Apache-2.0 / MS-PL |
| Python | none at runtime — the binding is `ctypes` from the standard library | — |

## Provenance of the C bridge

`core/src/llamabridge.cpp` and `core/include/llamabridge.h` were extracted from
[mochallama](https://github.com/deemwar-products/mochallama), MIT, by the same author.
See [ADR-0005](docs/adr/0005-mochallama-becomes-a-consumer.md).

## Models are never redistributed

No model weights ship in any artifact. The GGUF files used by the test suite are
downloaded by the developer at test time, and the tests **skip** when they are absent —
they are not vendored, not cached in the repo, and not published. Their licences
(Apache-2.0 for the Qwen2.5 and BGE families used in testing) are between the user and
the model publisher.

The `natives-*` release archives contain only compiled libraries and these notices.
