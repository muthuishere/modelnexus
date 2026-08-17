# Tasks — bundle the natives everywhere

## 1. Build produces a complete, self-describing closure

- [x] 1.1 `core/build.sh` writes `links.json` into `dist/<platform>/` — every symlink with its
      target, generated from what was actually staged
- [x] 1.2 `core/build.cmd` writes the same file (empty `links` on Windows); the shape must not
      differ by platform
- [x] 1.3 `core/build.sh --platform <key>` cross-builds, setting `CMAKE_OSX_ARCHITECTURES` and
      the llama.cpp asset name from the key rather than from `uname`
- [x] 1.4 A staged closure with a link whose target is absent fails the build — a closure that
      cannot load must not reach a release
- [ ] 1.5 Verify the npm platform tarballs carry symlinks (tar can; measured that wheels
      cannot, and dereference — 14 MB staged becomes 42 MB installed). If npm dereferences
      too, say so in the docs rather than quietly shipping 3x

## 2. darwin-x86_64 is built, not seeded

- [x] 2.1 Add `darwin-x86_64` to the `natives.yml` matrix on `macos-latest`, via `--platform`
- [x] 2.2 CI runs the cross-built native under Rosetta and requires a correct generated token.
      Architecture inspection alone is not acceptance — the Windows ARM native passed that and
      died in ggml's kernels
- [x] 2.3 Remove the hand-seeding one-liner from `PUBLISHING.md` and the "deliberately absent"
      note from `natives.yml:51`
- [x] 2.4 `release.yml` fails on a supported platform with no staged closure, instead of
      `skipping $plat (not staged)` — a silent skip is how an empty package ships

## 3. Go — the natives module

- [x] 3.1 New module `github.com/muthuishere/modelnexus/natives`, per-platform `//go:embed`
      behind build tags so the binary carries one platform and the module carries all
- [x] 3.2 Extraction: temp-then-rename, `.complete` stamp, content-hash cache key, `0o755`,
      lost-race-is-success. Spike 0009's `extract/main.go` is the reference, rewritten
- [x] 3.3 Replay `links.json`; fail loudly naming the entry if a link cannot be created
- [x] 3.4 `init()` registers the resolver at step 2, using the same hook shape as
      `RegisterScheme`
- [x] 3.5 `bindings/go` resolution order becomes the four ordered steps; the typed error names
      every step tried and why each missed
- [x] 3.6 Confirm `bindings/go`'s `go.mod` gains no dependency on the natives module

## 3b. Release the natives module

- [x] 3b.1 `release.yml` stages all five closures and pushes `natives/vX.Y.Z`
- [x] 3b.2 The payload commit is made on a DETACHED HEAD and only the tag is pushed, so `main`
      never carries ~70 MB per llama.cpp bump
- [x] 3b.3 The `replace` directive is stripped and the binding pinned to the released version;
      the job fails if a `replace` survives
- [x] 3b.4 `stage.sh --require-all` refuses a partial set — a module published with four of
      five closures fails at import on the fifth and nothing earlier would say so
- [x] 3b.5 `publish-go-natives` runs after `publish-go`, since it requires that tag to resolve

## 4. Python and JS reach the steps they are missing

- [ ] 4.1 Python honours `MODELNEXUS_LIB` first
- [ ] 4.2 Python falls back to `core/dist/<platform>/`, then to a `Fetch()` port
- [ ] 4.3 JS honours `MODELNEXUS_LIB` first
- [ ] 4.4 JS falls back to `core/dist/<platform>/`, then to a `Fetch()` port
- [ ] 4.5 Both raise a typed error naming every step tried when all four miss
- [ ] 4.6 C# follows the same order — unpublished, but ADR-0002 does not exempt it

## 5. Gate

- [ ] 5.1 Parity case: with `MODELNEXUS_LIB` unset and a bundled closure present, every binding
      reports the same resolved directory and the same `Version()`
- [ ] 5.2 Parity case: a bundled binding makes no network request while loading
- [ ] 5.3 Parity case: `MODELNEXUS_LIB` overrides the bundle in every binding
- [ ] 5.4 Regression: a closure materialised without its symlinks fails to load. This is the
      bug spike 0009 found; without a test it comes back the next time the payload changes

## 6. Docs

- [ ] 6.1 Document the four-step order once, in the shared docs, not three times per binding
- [ ] 6.2 Go README: the one-line opt-in and what the ~70 MB buys
- [x] 6.3 CHANGELOG under `## Unreleased`, written as what a user gets
- [ ] 6.4 State plainly that "bundled" means no network, not no filesystem — a read-only home
      directory is a real failure mode and the error must name the directory it tried

## Deferred

- **`linux-aarch64` does not exist** in the matrix or the release loop, though ADR-0007's
  context claims it does. Most cloud instances are ARM Linux now. Needs its own ADR; adding a
  supported platform is not a packaging decision.
- **`dlopen` from `memfd_create` on Linux**, removing the filesystem dependency for that
  platform only. A divergent load path for one OS is a poor trade. Recorded in `design.md` so
  it is not rediscovered as a bright idea.
