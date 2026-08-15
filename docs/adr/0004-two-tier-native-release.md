# ADR-0004 — Two-tier native release; nothing published from a laptop

- **Status:** accepted
- **Date:** 2026-08-15

## Context

Shipping a native library to five language registries means the cross product of platforms and
package managers. Building the native closure on every tag is slow, flaky, and — for
darwin-x86_64, which has no CI runner — impossible.

mochallama already solved this and runs it in production: a rare **tier 1** workflow builds
the per-platform native closure into a durable prerelease keyed by the llama.cpp tag
(`natives-b9371`), triggered only by changes under the C sources; a per-tag **tier 2**
workflow **never compiles** — it downloads the staged natives and publishes. Platforms without
a runner are seeded once from a laptop using `gh` only, never a registry credential.

## Decision

Adopt the same two-tier split.

- **Tier 1 (rare):** compile the native closure per platform on changes to `core/`. Output is
  a durable, tag-keyed prerelease of prebuilt libraries.
- **Tier 2 (every version tag):** compile nothing. Download the staged natives, then publish
  to Maven Central, PyPI, npm, NuGet, and the Go module tag.
- **Nothing is published from a developer machine.** Where a platform lacks a runner, the
  *artifact* may be staged locally via `gh`, but the publish is always CI.
- Prefer **OIDC trusted publishing** wherever the registry supports it (npm, PyPI, NuGet).
  Stored secrets only where no OIDC path exists.
- **All bindings are versioned and released together**, one tag, one version, every registry.
  A preflight job fails the run if any manifest disagrees with the tag.

## Consequences

- Tags are cheap and fast; a release is minutes, not an hour of native builds.
- The pinned llama.cpp tag becomes an explicit, visible part of the release identity.
- A native fix requires two workflow runs (restage, then tag). Accepted — it is rare.
- Verification is required *from outside*: install the published artifact from each registry
  in a clean directory and run it. Registry publishes have been observed to report success
  without landing; a green CI job is not evidence that a package is installable.

## Alternatives rejected

- **Build everything on every tag.** Slow, flaky, and unable to cover darwin-x86_64 at all.
- **Vendor natives in the repo.** Bloats clones, and makes the llama.cpp bump a source-control
  event rather than a build event.
