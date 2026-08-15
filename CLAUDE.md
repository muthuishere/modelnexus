# CLAUDE.md — modelnexus

Guidance for Claude Code working in this repo.

## What we're building

**modelnexus** — local LLM inference exposed as a **C ABI**, with one thin FFI binding per
language (Java/Panama, Go/purego, Python/ctypes, C#/P-Invoke, JS/koffi). llama.cpp
underneath, GGUF models, in-process, no daemon.

The C core is being extracted from [mochallama](https://github.com/deemwar-products/mochallama)'s
`llamabridge` (`core/src/main/cpp/`, ~890 LOC across header + impl), which has shipped to
Maven Central and npm. The header already declares a pure C surface with opaque handles and
JSON-in/JSON-out — **extraction is packaging work, not redesign** (see
`spikes/0001-bridge-portability/`).

## The prime directive: the ABI is the product

Everything else is a consumer. Two rules:

1. **The C surface is the contract.** A binding must never add behaviour, reinterpret a
   result, or work around a core bug locally. If a language needs something, it goes in the
   ABI, in the spec, and then in every binding — or it does not exist.
2. **One core, one runtime.** llama.cpp only. Never link a second inference engine into this
   core (ADR-0001). If ONNX in-process is ever needed, it is a *separate* core reusing the
   same distribution machinery.

The ABI is versioned and consumers link it. A breaking change here breaks applications, not
someone's afternoon — treat it with the discipline that implies.

## House workflow — ADR → spike → OpenSpec → code

Nothing of substance is written before it is decided, de-risked, and specified.

1. **ADR** (`docs/adr/NNNN-<slug>.md`) — a decision with lasting consequence: what we chose,
   what we rejected, why, and what it costs. Numbered, immutable once merged; a reversal is a
   *new* ADR that supersedes the old one, never an edit. Start here when the question is
   "which way do we go".
2. **Spike** (`spikes/NNNN-<slug>/`) — throwaway code that answers a factual question an ADR
   depends on ("can purego call this signature?", "does it cross-compile?"). A spike has a
   README stating the question, the method, and the **verdict**. Spikes are never promoted to
   production code — they are evidence, then they stop mattering. Prefer a spike over an
   argument.
3. **OpenSpec** (`openspec/changes/<name>/`) — `proposal.md`, `design.md`, `tasks.md`, and
   spec deltas under `specs/<capability>/spec.md`. This is where behaviour is pinned before
   it is coded. Slash commands: `/opsx:propose`, `/opsx:apply`, `/opsx:archive`. Validate
   with `openspec validate <name>`.
4. **Code** — implement the tasks, tick them off, run the suites, open the PR with the change
   folder and the code in one diff. **Archive only after merge.**

### Spec delta format (the part that bites)

- Requirement: `### Requirement: <name>`, SHALL/MUST wording.
- Scenario: **exactly four** hashes — `#### Scenario: <name>` — with `- **WHEN** …` / `- **THEN** …`.
  Three hashes or bullets fail silently at archive time.
- Every requirement needs ≥ 1 scenario. A `MODIFIED` requirement pastes the **full** updated
  block, not a fragment.

## Parity across bindings

A capability lands in the ABI **and every binding**, or it is not done — the same rule
toolnexus lives by. If a pass covers only some bindings, the rest stay as unchecked tasks in
`tasks.md`. Silent drift between bindings is the bug this repo exists to prevent.

Each binding is **idiomatic in its language** — not a transliteration. Same behaviour, native
shape: a Go binding returns `(T, error)`, a Java binding throws, a Python binding raises.

## Coding conventions

- The C surface stays **C**: no C++ types cross it, no name mangling, opaque handles only,
  strings as malloc'd UTF-8 released by an explicit `*_string_free`.
- Ownership must be obvious at the call site. Never return a pointer whose lifetime rule
  isn't documented in the header.
- Bindings hold no state the core doesn't; they marshal and they free.
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `ci:`, scoped by binding
  where it helps (`feat(go): …`). Do **not** add `Co-authored-by:`.
- Touch only what the task needs; note unrelated issues rather than fixing them inline.

## CHANGELOG.md

Every change a user could notice adds to `## Unreleased` **in the same PR**. Write what a
user gets, not what a file did. Name what is *not* done and where it is tracked. On release,
`## Unreleased` becomes `## X.Y.Z — YYYY-MM-DD`.

## Releasing

Not yet wired. The intended model is mochallama's proven two-tier split (ADR-0004): a rare
native-build tier producing durable per-platform prereleases, and a per-tag publish tier that
**never compiles** — it downloads the natives and pushes to the registries. Nothing is ever
published from a laptop.

## Do not

- Do not link a second inference runtime into the core.
- Do not let the core grow a tool executor, an agent loop, or a retrieval layer.
- Do not add a dependency from modelnexus to toolnexus, or the reverse.
- Do not tag or publish unless the owner asked.
