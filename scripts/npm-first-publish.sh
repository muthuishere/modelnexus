#!/usr/bin/env bash
#
# ONE-TIME BOOTSTRAP: create the npm packages from a logged-in machine, so that
# trusted publishing (OIDC) can take over for every release afterwards.
#
# WHY THIS EXISTS, AND WHY IT IS NOT THE NORMAL PATH
#
# ADR-0004 says nothing is published from a laptop, and that stands. This is the
# single case the rule cannot cover: npm's trusted publishing is configured on a
# package's settings page, so it REQUIRES THE PACKAGE TO ALREADY EXIST. There is no
# pending-publisher equivalent (PyPI has one; npm does not). A registry with nothing
# in it cannot be published to by OIDC, so something has to create the packages once.
#
# The alternative is putting a long-lived automation token in GitHub secrets. This is
# better: the credential never leaves the machine that already has it, and after this
# runs there is no credential anywhere to leak.
#
# AFTER THIS SUCCEEDS:
#   1. Configure a trusted publisher on each of the six packages
#      (Settings -> Trusted publisher: muthuishere/modelnexus, workflow release.yml)
#   2. Every later release goes through release.yml with no secret at all.
#   3. Never run this script again.
#
# It reproduces .github/workflows/release.yml's publish-npm step exactly -- same
# staged closures, same package layout, same dereferencing copy. If you change one,
# change the other.
#
#   ./scripts/npm-first-publish.sh 0.2.1              # DRY RUN: builds and reports
#   ./scripts/npm-first-publish.sh 0.2.1 --publish    # actually publishes
#
set -euo pipefail

VERSION="${1:-}"
PUBLISH=0
for a in "${@:2}"; do [ "$a" = "--publish" ] && PUBLISH=1; done
[ -n "$VERSION" ] || { echo "usage: $0 <version> [--publish]" >&2; exit 2; }

cd "$(dirname "$0")/.."
ROOT="$PWD"
WORK="$ROOT/.npm-first-publish"
PLATFORMS="linux-x86_64 darwin-aarch64 darwin-x86_64 windows-x86_64 windows-aarch64"

# ---- preflight --------------------------------------------------------------
who="$(npm whoami 2>/dev/null || true)"
[ -n "$who" ] || { echo "not logged in to npm. Run: npm login" >&2; exit 1; }
echo "==> npm user: $who"

js_version="$(node -p "require('./bindings/js/package.json').version")"
[ "$js_version" = "$VERSION" ] || {
  echo "bindings/js/package.json is $js_version, you asked for $VERSION." >&2
  echo "One version everywhere, or no release -- same rule release.yml enforces." >&2
  exit 1
}

llama_tag="$(bash core/build.sh --print-tag)"
echo "==> version $VERSION, llama.cpp $llama_tag"

# ---- the same closures the workflow would publish ---------------------------
rm -rf "$WORK"; mkdir -p "$WORK/zips" "$WORK/staged"
echo "==> downloading staged natives (never built here -- ADR-0004 tier 1 owns that)"
gh release download "natives-$llama_tag" --pattern '*.zip' --dir "$WORK/zips"
for z in "$WORK"/zips/*.zip; do unzip -qo "$z" -d "$WORK/staged"; done

missing=""
for plat in $PLATFORMS; do
  [ -f "$WORK/staged/$plat/links.json" ] || missing="$missing $plat"
done
[ -z "$missing" ] || {
  echo "not staged:$missing" >&2
  echo "Run the natives workflow for the current pin first. A package with no library" >&2
  echo "in it installs cleanly and fails at import, and npm versions are immutable." >&2
  exit 1
}
echo "==> all five platforms staged"

# ---- build the packages -----------------------------------------------------
for plat in $PLATFORMS; do
  case "$plat" in
    linux-x86_64)    os=linux;  cpu=x64 ;;
    darwin-aarch64)  os=darwin; cpu=arm64 ;;
    darwin-x86_64)   os=darwin; cpu=x64 ;;
    windows-x86_64)  os=win32;  cpu=x64 ;;
    windows-aarch64) os=win32;  cpu=arm64 ;;
  esac
  pkg="$WORK/pkg-$plat"
  mkdir -p "$pkg/native/$plat"
  # -RL, NOT -a. npm pack silently DROPS symlinks, and the bridge links the versioned
  # names (@rpath/libllama.0.dylib) which ARE symlinks -- a link-preserving copy
  # produces a package that installs cleanly and fails at first use. Verified in CI:
  # 29 files became 11 and both linked libraries went missing.
  cp -RL "$WORK/staged/$plat"/. "$pkg/native/$plat/"
  cat > "$pkg/package.json" <<JSON
{
  "name": "@muthuishere/modelnexus-$plat",
  "version": "$VERSION",
  "description": "modelnexus native library for $plat",
  "os": ["$os"],
  "cpu": ["$cpu"],
  "files": ["native"],
  "license": "MIT"
}
JSON
  echo "  built @muthuishere/modelnexus-$plat ($(du -sh "$pkg/native" | cut -f1))"
done

# The launcher pins its optionalDependencies by version, from the one source of truth.
# A hand-maintained pin drifts the moment someone bumps package.json, and the failure
# is silent: npm skips an unresolvable OPTIONAL dependency without a word.
node -e '
  const fs = require("fs");
  const p = "./bindings/js/package.json";
  const j = JSON.parse(fs.readFileSync(p, "utf8"));
  for (const k of Object.keys(j.optionalDependencies || {})) j.optionalDependencies[k] = process.argv[1];
  fs.writeFileSync(p, JSON.stringify(j, null, 2) + "\n");
  console.log("  pinned launcher optionalDependencies to " + process.argv[1]);
' "$VERSION"

# ---- publish ----------------------------------------------------------------
# NPM_CONFIG_PROVENANCE is deliberately NOT set: provenance requires OIDC in a CI
# runner and fails from a laptop. The CI path sets it; this one cannot, and saying so
# is better than a confusing failure. It is one more reason this runs exactly once.
if [ "$PUBLISH" != 1 ]; then
  echo
  echo "DRY RUN. Nothing was published. Would publish, as $who:"
  for plat in $PLATFORMS; do echo "  @muthuishere/modelnexus-$plat@$VERSION"; done
  echo "  @muthuishere/modelnexus@$VERSION"
  echo
  echo "npm versions are IMMUTABLE -- a mistake here cannot be republished, only"
  echo "deprecated. Check the list, then re-run with --publish."
  exit 0
fi

publish_dir() {
  local dir="$1" out
  # Never `|| true`: that swallows an auth failure and reports success while
  # publishing nothing. CI did exactly that once -- green job, four packages still
  # 404. Only a genuine "already published" is tolerated.
  if out="$(npm publish "$dir" --access public 2>&1)"; then
    echo "$out" | tail -1
    return 0
  fi
  echo "$out"
  if grep -qiE "cannot publish over|EPUBLISHCONFLICT|previously published|409" <<<"$out"; then
    echo "  -> already published at this version; continuing"
    return 0
  fi
  return 1
}

for plat in $PLATFORMS; do publish_dir "$WORK/pkg-$plat"; done
publish_dir "$ROOT/bindings/js"

echo
echo "==> published. NOW DO THIS, or the next release has no way in:"
echo "    configure a trusted publisher on all six packages"
echo "    (repo muthuishere/modelnexus, workflow release.yml), then never run this again."
