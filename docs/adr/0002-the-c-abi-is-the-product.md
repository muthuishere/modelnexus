# ADR-0002 — The C ABI is the product; bindings are thin

- **Status:** accepted
- **Date:** 2026-08-15

## Context

The alternative to a shared C core is to implement inference access natively per language —
five codebases, five sets of bugs, five interpretations of the same llama.cpp behaviour. That
is the drift toolnexus spends a byte-parity conformance gate to prevent, and it would be far
worse here because the behaviour being duplicated is native and stateful.

mochallama already proved the other path: a ~890-LOC `extern "C"` surface (header + impl) over
llama.cpp's `common_chat`, consumed from Java via Panama. Spike 0001 found the header carries
no Java-specific coupling — opaque handles, JSON in, JSON out, malloc'd UTF-8 with an explicit
free. It was written to be bound "from any FFI" and it is.

## Decision

The **C ABI is the deliverable**. Each language gets a thin binding whose only jobs are:
marshal arguments, call, free, and present the result in the language's idiom.

A binding **must not**: add behaviour, reinterpret or repair a core result, cache state the
core doesn't, or locally work around a core bug. If a language needs something, it goes into
the ABI, into the spec, and then into every binding — or it does not exist.

Bindings are **idiomatic, not transliterated**. Go returns `(T, error)`, Java throws, Python
raises, C# uses `IDisposable`, JS returns promises. Same behaviour, native shape.

## Consequences

- One place to fix an inference bug; one place to add a capability.
- A binding is small enough to be written and reviewed in days, which is what makes five
  languages affordable for one maintainer.
- The ABI becomes a versioned public contract. Breaking it breaks *applications*, so it needs
  release discipline closer to a runtime library than to a CLI — see ADR-0004.
- Capability parity across bindings must be tracked explicitly, the way toolnexus tracks port
  parity, or bindings silently diverge in coverage.

## Alternatives rejected

- **Per-language native implementations.** Five copies of the hardest code in the stack.
- **A local HTTP server plus thin clients.** Simpler to build, but forfeits the entire
  proposition: in-process, no daemon, no port, no supervision, no data crossing a socket. A
  server remains available *on top* of the core for anyone who wants one.
