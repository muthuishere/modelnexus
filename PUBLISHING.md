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

## Status of 0.2.0

**Nothing is publishable until the natives are re-staged.** The assets on `natives-b9371` are
the **0.1.0** bridge — 14 symbols. 0.2.0 adds `llb_count_tokens`, `llb_chat_cache` and
`llb_last_error`, and a published binding against those assets fails at load with a missing
symbol. Verified by downloading the staged asset and reading its symbol table.

`natives.yml` runs on a push to `main` touching `core/**`, so merging the change re-stages them.
It clobbers the same `natives-b9371` release, which is safe here because 0.2.0's exports are a
**superset** of 0.1.0's — an existing 0.1.0 consumer that fetches afterwards still finds every
symbol it needs. That is luck, not design: the natives release is keyed on the llama.cpp tag
alone (ADR-0004) while the bridge moves independently, so a future release that *removes* a
symbol would break old consumers silently. The Go client cache is now keyed on both, but the
release itself is not.

Released 2026-08-16. Natives re-staged first and **verified by reading the asset's symbol
table**, not by trusting a green check: 17 symbols, including the three 0.2.0 additions.

| leg | state |
|---|---|
| **Go** | **PUBLISHED and verified from outside.** `go get github.com/muthuishere/modelnexus/bindings/go@v0.2.0` from the public proxy into a clean module; with `MODELNEXUS_LIB` unset, `Fetch()` resolved its own cache and `CountTokens` — a 0.2.0-only entry point — returned `35 tokens of 4096`. That last part is deliberate: it cannot pass against stale natives. |
| **npm** | Failed: `npm error code ENEEDAUTH`. |
| **PyPI** | Failed: no publisher registered. |

`ENABLE_GO`, `ENABLE_NPM`, `ENABLE_PYPI` are all `true`. The repo has **zero secrets**, which is
the whole of the npm/PyPI failure — see the owner-only setup below. Re-running after adding
either credential is safe: every leg is idempotent.

### The docs site had never been enabled

`pages.yml` built 18 pages and failed at deploy with `HttpError: Not Found` — GitHub Pages was
never turned on for the repo. Enabled with `build_type: workflow`; the site is live at
<https://muthuishere.github.io/modelnexus/>.

Worth naming, because it is the same failure shape as the stale natives: the build was green and
the artifact was correct. It simply had nowhere to go. Read what reached the user, never what the
job said.

## Status of 0.1.0

| leg | state |
|---|---|
| **Go** | **PUBLISHED and verified.** `go get github.com/muthuishere/modelnexus/bindings/go@v0.1.0` from the public proxy, `Fetch()` pulls natives from the `natives-b9371` release, inference runs. Needs no credential — it is a tag push using `github.token`. |
| **npm** | Blocked: `npm error code ENEEDAUTH`. |
| **PyPI** | Blocked: `invalid-publisher: valid token, but no corresponding publisher`. The OIDC token is fine; PyPI has nothing registered to match it. |

Natives are staged for **linux-x86_64, darwin-aarch64, windows-x86_64** at the
`natives-b9371` release, all verified as real zips containing the bridge.

Re-running after you add credentials is safe: `gh workflow run release.yml -f version=0.1.0`.
The Go leg is idempotent (skips an existing tag), PyPI uses `skip-existing`, and npm tolerates
a genuine already-published conflict while failing on anything else.

## Owner-only setup (blocks the first publish)

These are web-UI actions on the registries. They cannot be done from the CLI, and until they
exist the PyPI and npm jobs fail on authentication.

### 1. PyPI — pending trusted publisher

The distribution is **`muthuishere-modelnexus`**, not `modelnexus`: that name is held by an unrelated
project ("Global AI Model Vault", 1.0.4) and PyPI normalises names case-insensitively. The
import is unchanged — `import modelnexus`.

Because the project does not exist yet, it needs a **pending** publisher:

> pypi.org → Your account → Publishing → *Add a new pending publisher*
> - PyPI Project Name: `muthuishere-modelnexus`
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
