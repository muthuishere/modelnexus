# ADR-0006 — modelnexus ships no Java binding

- **Status:** accepted
- **Date:** 2026-08-15
- **Supersedes:** the Java-binding clause of [ADR-0005](0005-mochallama-becomes-a-consumer.md)

## Context

ADR-0005 said the Java/Panama binding would move from mochallama into modelnexus, so
that all five languages came off one surface.

Two things changed that. First, the owner's read of the audience: the Java ecosystem
has not returned interest proportional to the effort — the demand is in Go, Python, C#
and JavaScript. Second, and more concretely, mochallama's Panama binding **already
exists, already ships to Maven Central, and already works**. Moving it would be pure
migration cost against a language nobody is asking for.

## Decision

modelnexus ships bindings for **Go, Python, C# and JavaScript**. There is no Java
binding here.

mochallama keeps its own Panama binding in-tree and continues to ship as it does
today. It tracks the core by tag rather than by dependency — for now, that means the
two carry compatible copies of the same C surface, and mochallama remains the JVM
answer for anyone who wants one.

The rest of ADR-0005 stands: the core is owned here, mochallama is a consumer of the
*surface* rather than its owner, and its published API does not change.

## Consequences

- Four bindings to keep at parity instead of five. For a single maintainer that is a
  material, permanent reduction in cost.
- Java users are served by mochallama, which is the better experience for them anyway
  — Spring autoconfiguration, an OpenAI-compatible endpoint, Actuator, a Spring AI
  adapter. A raw FFI binding was never the thing a Spring developer wanted.
- The two copies of the C surface can drift. That is a real risk and it is accepted
  deliberately: mochallama pins its own llama.cpp tag and is free to lag. If it ever
  becomes a problem, the fix is for mochallama to consume the published core, which is
  what ADR-0005 always envisaged.
- Spike 0002's callback-lifetime finding remains **unverified for Panama**. It does not
  matter here, but mochallama should confirm it independently — the core retains the
  event callback, and a JVM binding can hit the same crash.

## Alternatives rejected

- **Port the Java binding anyway, for completeness.** Completeness is not a user.
- **Drop mochallama too.** It is live on Maven Central and npm with real users, and it
  is the only in-process JVM option of its kind. Retiring a working, published product
  to tidy a diagram is a bad trade.
