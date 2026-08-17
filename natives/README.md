# modelnexus/natives

The native library, bundled. Add one import and modelnexus never touches the network to
load it:

```go
import _ "github.com/muthuishere/modelnexus/natives"
```

Nothing else changes. Your existing code keeps working; `Open`, `Version`, everything.

## Why this is a separate module

Go has no optional dependencies, and `go:embed` selects at compile time rather than fetch
time. Putting five platforms inside `bindings/go` would put **~70 MB in the module cache of
every consumer** — including the ones who set `MODELNEXUS_LIB` and never wanted a bundle.

A build tag would not help: a tagged file's imports still land in `go.mod`. That is the same
constraint that put the S3 model resolver in its own module.

So the cost is opt-in. Import this and you pay ~70 MB once, in the module cache. Skip it and
`bindings/go` stays a source tree with one dependency.

**The binary is not 70 MB.** The embeds are behind build tags, so an executable carries only
its own platform — measured at ~20 MB for a trivial program on darwin-aarch64, of which 14 MB
is the closure.

## What "bundled" means

**No network.** It does not mean no filesystem. A shared library cannot be portably loaded
from memory, so the closure is written to your user cache directory (or `MODELNEXUS_CACHE`)
before it is opened. That happens once; later starts find it and skip straight to loading.

A read-only home directory is therefore a real failure mode. It produces a typed error naming
the directory it tried, not a mystery.

## Where this sits in resolution

The binding tries, in order, stopping at the first that works:

1. `MODELNEXUS_LIB` — an explicit path always wins. **A bundle never takes that away.**
2. **this bundle**, if you imported it
3. the repository's `core/dist/<platform>/`, for working in the tree
4. a download from the tag-keyed GitHub release

Ahead of 3 and 4 on purpose: importing this module is a request for a hermetic build, and
reaching the network afterwards would be the opposite of what you asked for.

## Developing on it

The payload is **not committed** — ~14 MB per platform × 5, on every llama.cpp bump, forever,
for everyone who clones the repo. `natives/payload/*/` holds a placeholder so `go:embed` has
something to match; a module built from a fresh clone compiles and then reports "native bridge
not found" at first use, which is the honest outcome for a bundle nobody populated.

```bash
../core/build.sh              # build a closure
./stage.sh                    # copy core/dist/* into payload/
go test ./...
```

`stage.sh` refuses a closure with no `links.json`, because that manifest is what makes the
bundle loadable at all — see below.

## The symlink problem, in case you touch the extraction code

`go:embed` takes **regular files only**. It does not record symbolic links and does not report
dropping them. A staged closure has 18 of them, and the bridge links `@rpath/libllama.0.dylib`
— which is one.

So a naive embed produces a directory that looks complete, weighs about right, and cannot
load. The build writes `links.json` beside the closure and extraction replays it. Do not
"simplify" that away; there is a regression test, and it exists because this bug was real.
