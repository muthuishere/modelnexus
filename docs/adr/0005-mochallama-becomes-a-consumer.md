# ADR-0005 — mochallama becomes a consumer of the core, not its owner

- **Status:** accepted
- **Date:** 2026-08-15

## Context

`llamabridge` lives inside mochallama's `core` Gradle subproject. That is an accident of
birth: the bridge was written for a Spring-first JVM product, so it was filed under it. The
surface itself is language-neutral (spike 0001), and four other languages want it.

mochallama is live on Maven Central and npm with real users. Extraction must not disturb them.

## Decision

The C core moves to modelnexus and becomes the shared artifact. **mochallama remains a
product** — the Spring-flavoured JVM experience: `@AutoConfiguration`, the OpenAI-compatible
endpoint, Actuator metrics, the Spring AI `ChatModel` adapter, the Picocli/`npx` CLI, the demo
app. It consumes the core instead of containing it.

The Java binding lives in **modelnexus**, not mochallama. mochallama depends on it.

Migration is staged and reversible: extract and prove the core with a second binding *first*,
switch mochallama to consume it *second*, and only then delete the vendored copy. At no point
is mochallama's published surface allowed to change.

## Consequences

- mochallama gets smaller and more clearly positioned: a Spring integration, not an inference
  engine.
- Its release process now has an upstream. Version pinning between the two must be explicit.
- Four other languages get everything mochallama's users already have.
- Short-term duplication is expected and accepted while both copies exist.

## Alternatives rejected

- **Leave the bridge in mochallama and have other languages depend on it.** Would make a
  Java/Gradle/Spring repository the upstream for a Go and Python library. The build tooling
  and release cadence are wrong for every non-JVM consumer.
- **Fork the bridge per language.** The exact drift this repo exists to prevent.
