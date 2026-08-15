# Publishing

How modelnexus reaches its registries, what is already wired, and the two things only the
account owner can do.

Publishing targets are **Go, PyPI and npm**. There is no NuGet leg — the C# binding works and
is tested, it is simply not published ([ADR-0007](docs/adr/0007-how-natives-reach-each-ecosystem.md)).

## The two-tier model

Nothing is ever published from a laptop ([ADR-0004](docs/adr/0004-two-tier-native-release.md)).

| workflow | when | what it does |
|---|---|---|
| `natives.yml` | changes under `core/**` | **The only thing that compiles.** Builds the native closure per platform into a durable prerelease keyed by the llama.cpp tag (`natives-b9371`). |
| `release.yml` | a published GitHub Release, or `workflow_dispatch` | **Never compiles.** Downloads the staged natives, packages per ecosystem, publishes. |

`release.yml` refuses to run if `natives-<tag>` is not staged, rather than publishing packages
with no library inside them.

**Do not delete the `natives-*` prerelease.** Go users resolve their runtime library from it at
first use via `modelnexus.Fetch()`, so it is a runtime dependency, not merely a build input.

## Owner-only setup (blocks the first publish)

These are web-UI actions on the registries. They cannot be done from the CLI, and until they
exist the PyPI and npm jobs fail on authentication.

### 1. PyPI — pending trusted publisher

The distribution is **`modelnexus-core`**, not `modelnexus`: that name is held by an unrelated
project ("Global AI Model Vault", 1.0.4) and PyPI normalises names case-insensitively. The
import is unchanged — `import modelnexus`.

Because the project does not exist yet, it needs a **pending** publisher:

> pypi.org → Your account → Publishing → *Add a new pending publisher*
> - PyPI Project Name: `modelnexus-core`
> - Owner: `muthuishere` · Repository: `modelnexus`
> - Workflow name: `release.yml`
> - Environment: *(leave blank)*

### 2. npm — trusted publisher, or a token

Five packages publish: the launcher `@muthuishere/modelnexus` plus one per platform
(`@muthuishere/modelnexus-<platform>`), wired together through `optionalDependencies` so npm
installs only the one that matches.

Either configure OIDC trusted publishing for each package on npmjs.com, **or** add an
automation token as a repo secret and reference it as `NODE_AUTH_TOKEN` in the `publish-npm`
job. The token path is the pragmatic one for a first publish, since trusted publishing is
easier to attach to a package that already exists.

### 3. Repo variables — already set

`ENABLE_GO`, `ENABLE_PYPI`, `ENABLE_NPM` are all `true`. A leg whose variable is false is
skipped **silently**, so read the job list: a green run with skipped legs is not a release.

## Cutting a release

1. Bump `bindings/js/package.json`, `bindings/python/pyproject.toml`, and
   `LLB_BRIDGE_VERSION` in `core/CMakeLists.txt` to the same version. Preflight fails the run
   if the two manifests disagree with the tag — that is the guard against a partial bump.
2. Move `## Unreleased` in `CHANGELOG.md` to `## X.Y.Z — YYYY-MM-DD`, fresh empty
   `## Unreleased` above it.
3. Ensure `natives.yml` has run for the current llama.cpp pin.
4. Publish a GitHub Release `vX.Y.Z`. **The body is the changelog section**, not a second
   account of the same work.
5. Read the job list. Confirm each enabled leg actually published.
6. **Verify from outside, as a consumer would** — install from the registry into a clean
   directory and run something. Registry publishes have been observed to report success
   without landing.

## What has been verified locally

Both packaging paths were built and installed as a consumer would, before any publish:

- **Wheel**: built, installed into a clean venv, ran inference and embeddings with no repo
  nearby and `MODELNEXUS_LIB` unset.
- **npm**: both tarballs packed, installed into a fresh project, ran inference and embeddings.

That exercise caught two bugs that would each have shipped a package that installs cleanly and
fails at first use:

- `npm pack` **silently drops symlinks**. llama.cpp ships versioned aliases as symlinks and the
  bridge links the versioned names, so the platform package went from 29 files to 11 with both
  linked libraries missing. Packaging now dereferences (`cp -RL`).
- `bindings/js` declared **no `optionalDependencies`**, and its loader only looked inside its
  own package — so the launcher could never find a separately published platform package.

## Platform notes

- **Windows builds from source.** llama.cpp publishes no `.lib` import libraries for Windows,
  and MSVC cannot link a DLL without one, so the prebuilt fast path is impossible there. Slow,
  and confined to `natives.yml`.
- **Intel macOS (`darwin-x86_64`) is NOT built in CI** — those runners queue indefinitely, and
  one job that never starts blocks publishing for every platform that did. It is absent from
  both matrices by design. `release.yml` skips platforms that are not staged, so the other three
  publish normally. To add Intel macOS, stage it by hand from an Intel Mac:
  `./core/build.sh && cd core/dist && zip -ry natives-darwin-x86_64.zip darwin-x86_64 &&
  gh release upload natives-b9371 natives-darwin-x86_64.zip --clobber`
  (this is the pattern mochallama already uses for the same platform).

## Not verified

- **No release has been cut**, so the workflows have not run end to end.
