# ADR-0007 — How the native library reaches each ecosystem

- **Status:** accepted
- **Date:** 2026-08-15

## Context

The bridge plus its llama.cpp runtime closure is **~14 MB per platform** (40 MB before
symlink preservation was fixed — llama.cpp ships versioned aliases, and dereferencing
them tripled the largest library). Five platforms: linux x64/arm64, darwin x64/arm64,
windows x64.

Each package manager has a different idea of how a native library should travel, and
the wrong answer in each is the same: make the user install a toolchain or hunt for a
`.so`. ADR-0002 says a binding marshals and frees; it should not also be a build system.

Publishing targets are **Go, Python and JavaScript**. C# has a working binding but is
not published — no NuGet leg.

## Decision

Different mechanisms per ecosystem, chosen to match how each one already works.

### Python — platform wheels

Natives ship inside the wheel at `modelnexus/native/<platform-key>/`. The user runs
`pip install modelnexus` and gets a working library, because that is exactly what a
wheel is for. One wheel per platform tag; the sdist carries no natives and fails
loudly rather than silently building nothing.

### JavaScript — optionalDependencies

One package per platform (`@muthuishere/modelnexus-darwin-arm64`, …), plus a launcher
package that lists them all in `optionalDependencies`. npm installs only the one that
matches, and the others are skipped without error. This is the pattern mochallama's
CLI already uses in production, so it is proven in this portfolio rather than merely
plausible.

### Go — NOT bundled

A Go module is a source tree. Shipping five platforms' binaries inside it makes every
`go get` download ~70 MB of libraries the user will not use, and `go:embed` cannot
select at fetch time — only at compile time, with the files already present.

So the Go binding **locates the library at runtime** and does not carry it:

1. `MODELNEXUS_LIB` — explicit path, wins over everything.
2. The repo's `core/dist/<platform>/`, for working in the tree.
3. `modelnexus.Fetch()` — downloads the platform's natives from the GitHub release
   matching the pinned llama.cpp tag into a user cache directory, once.

This is the same shape `onnxruntime_go` uses, and it is honest about the constraint
rather than pretending a Go module is a package manager for binaries.

## Consequences

- Python and JS users get "install and it works". Go users take one extra step, which
  is documented at the top of the Go README and produces a **typed error naming every
  path searched** when skipped — never a segfault.
- The natives are built once per llama.cpp tag (ADR-0004 tier 1) and reused by all
  three publishing legs. Nothing compiles on a release tag.
- The GitHub release holding the natives becomes a load-bearing artifact for Go users
  at runtime, not just at build time. It must not be deleted; the tag-keyed prerelease
  is durable for exactly this reason.
- A Go user behind a firewall with no `MODELNEXUS_LIB` and no cached download gets a
  clear failure, not a mystery. That is the best available outcome, and it is why the
  error lists what it looked for.

## Alternatives rejected

- **`go:embed` all platforms.** ~70 MB module for 14 MB of use.
- **cgo, so the library links normally.** Loses `CGO_ENABLED=0` and cross-compilation,
  which is most of why purego was chosen (ADR-0001 discussion, spike 0002).
- **Require every user to run `core/build.sh`.** Needs cmake and a compiler. That is
  the "native-install dance" this project exists to remove.
