# Design — extracting the core

## Where the seam goes

The extracted core is **exactly** the current C surface plus its build, and nothing else. The
Java-side `ChatEngine` / `MochallamaClient` conveniences stay in mochallama; the binding in
this repo is the raw, idiomatic FFI layer that they will later sit on.

## Symbol prefix

The shipped surface uses `llb_`. Options were: keep it, or rename to `mnx_`.

**Decision: keep `llb_` for this change.** Renaming during an extraction means every failure is
ambiguous — is it the move or the rename? A rename, if wanted, is a mechanical follow-up change
with a compatibility header. Repo naming and symbol naming need not agree.

## Build

Gradle owns mochallama's native build today. The standalone core must build without it, since
Go and Python consumers have no reason to install a JVM toolchain. Target: CMake, producing a
shared library per platform, consuming **prebuilt llama.cpp release libraries** — the mode
mochallama already defaults to. We compile the bridge, never llama.cpp. That is what keeps the
native build to seconds instead of an hour, and it is already proven.

The llama.cpp tag is pinned and is part of the release identity (ADR-0004).

## The Go binding

`purego` calls the shared library with no cgo, so `CGO_ENABLED=0` stays valid and
cross-compilation is preserved. The library is located at runtime; the binding must therefore
have an explicit, well-typed failure for "runtime not found / not initialized" — a failure mode
the JVM, Python and .NET bindings get for free from their package managers and Go does not.
This is a genuine behavioural difference between bindings and it is specified, not left to
each binding to invent.

The same purego load seam is already proven across linux x64+arm64, macOS arm64 and windows x64
in citenexus (`spikes/prebuilt-ffi/`). Reuse that evidence rather than re-deriving it.

## Callbacks across the boundary

`llb_event_cb` and `llb_token_cb` are plain C function pointers invoked from native code,
potentially on a non-caller thread. Each runtime reacts differently — goroutine scheduling,
the GIL, .NET's GC moving delegates. Spike 0002 answers this per language **before**
`llb_chat_infer_stream` is bound. If a runtime cannot take the callback safely, the fallback is
a poll-based drain in that binding only, and that divergence gets specified.

## Errors

The core never returns NULL for inference — failures come back as error JSON with a `code` and
a `message`. Each binding maps that onto its own idiom: Go `(T, error)`, Java exception, Python
raise. The **codes are part of the contract** and must be identical everywhere; the surface
that carries them is not.
