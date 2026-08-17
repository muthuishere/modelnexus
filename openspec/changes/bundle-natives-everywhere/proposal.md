# Bundle the natives in every ecosystem

## Why

Three bindings, three different answers to "where is the shared library", and only one of
them is written down as a contract.

Python and JavaScript bundle: `pip install` and `npm install` produce something that works.
Go downloads at runtime, which ADR-0007 justified on size — five platforms in one Go module
is ~70 MB pulled by everyone.

The size argument is still true. What was underweighted is everything on the other side:

- **Failure moves to run-time.** `pip install` succeeds and `import modelnexus` reaches for
  the network the first time it runs — in a container, in a Lambda, at 3am. Install-time
  failure happens while a human is watching.
- **Docker caching breaks.** The install layer no longer contains the library, so it cannot
  be cached. Every fresh container pays again.
- **Offline installs break.** `pip download` on a connected box, copy, `pip install
  --no-index` — the workflow of exactly the enterprise that wants an in-process model and no
  daemon.
- **Provenance stops covering the binary.** `NPM_CONFIG_PROVENANCE` and PyPI attestations sign
  the published artifact. A library fetched at startup is outside that chain.
- **A GitHub release becomes load-bearing at runtime for every user**, subject to
  unauthenticated rate limits that a CI fleet behind one NAT will hit.

ADR-0010 settles it: **bundle, and keep downloading as the fallback.** This change implements
that, and closes two platform holes that bundling turns from embarrassing into fatal.

## What changes

**Go gains an opt-in `natives` module.** `import _ ".../natives"` and there is no network,
ever. Not in `bindings/go`, because Go has no optional dependencies and no fetch-time
selection — the 70 MB would be paid by everyone, including consumers who only set
`MODELNEXUS_LIB`. Separate module, same reasoning that already put the S3 resolver in one
(`source.go:199`).

**All three bindings get the same four-step resolution order**, in the same order, proven by
the parity gate:

1. `MODELNEXUS_LIB` — explicit path, wins
2. the bundled closure — no network
3. `core/dist/<platform>/` — working in the tree
4. `Fetch()` — download from the tag-keyed release

Today Go has 1, 3, 4 and Python/JS have 2. Afterwards everyone has all four. This is the part
that makes it a specification rather than three implementations that happen to work.

**The build emits a symlink manifest.** Spike 0009 found that `go:embed` silently drops all 18
symlinks in `dist/`, including `libllama.0.dylib` — which is precisely what the bridge links
via `@rpath`. The result looks complete, weighs about right, and cannot load. Dereferencing
instead would take ~14 MB to ~40 MB per platform. So `build.sh` and `build.cmd` generate
`links.json`, and extraction replays it.

**`darwin-x86_64` joins the build matrix.** It is currently seeded by hand, which in practice
means absent — `natives-b9371` has four assets. Bundling makes an absent platform fail at
import rather than at install. Spike 0009 shows no Intel runner is needed: `build.sh` never
compiles llama.cpp, so `-DCMAKE_OSX_ARCHITECTURES=x86_64` on the arm64 runner produces a
native that loads and generates correctly under Rosetta.

## What does not change

- The C ABI. Nothing here crosses it.
- The existing Go API. A consumer who does not add the import behaves exactly as today.
- Python and JS packaging mechanics — wheels and `optionalDependencies` were already right.
- `Fetch()` stays. It is the escape hatch for an unbuilt platform, a different llama.cpp pin,
  and an air-gapped mirror.

## Out of scope

**`linux-aarch64` does not exist anywhere** — not in the matrix, not in the release loop —
although ADR-0007's context claims it does. ARM Linux is most cloud instances now, so this is
a real gap, but adding a platform to the supported set is its own decision and deserves its
own ADR rather than arriving as a matrix row in a packaging change.
