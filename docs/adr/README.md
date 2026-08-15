# Architecture Decision Records

One file per decision with lasting consequence. Numbered, dated, and **immutable once
merged** — a reversal is a new ADR that supersedes the old one, never an edit to it. If you
find yourself re-arguing a settled question, the answer is either "read the ADR" or "write
the superseding one".

Format: Context → Decision → Consequences → Alternatives rejected. Keep the rejected
alternatives — six months from now that section is the whole value of the document.

| # | title | status |
|---|---|---|
| [0001](0001-one-core-one-runtime.md) | One core, one runtime — llama.cpp only | accepted |
| [0002](0002-the-c-abi-is-the-product.md) | The C ABI is the product; bindings are thin | accepted |
| [0003](0003-modelnexus-and-toolnexus-do-not-depend-on-each-other.md) | modelnexus and toolnexus do not depend on each other | accepted |
| [0004](0004-two-tier-native-release.md) | Two-tier native release; nothing published from a laptop | accepted |
| [0005](0005-mochallama-becomes-a-consumer.md) | mochallama becomes a consumer of the core, not its owner | accepted (Java clause superseded by 0006) |
| [0006](0006-no-java-binding.md) | modelnexus ships no Java binding | accepted |
