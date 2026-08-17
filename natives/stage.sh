#!/usr/bin/env bash
#
# Copy staged native closures into the natives module's payload directories.
#
#   ./stage.sh                    from core/dist/ (whatever you have built locally)
#   ./stage.sh path/to/staged     from a directory of <platform-key>/ trees
#   ./stage.sh DIR --require-all  fail unless EVERY platform lands. What a release
#                                 uses: a module published with four of five
#                                 closures is a module that fails at import on the
#                                 fifth, and nothing before this point would say so.
#
# The payload is NOT committed. Five platforms at ~14 MB each is ~70 MB, and a
# fresh copy on every llama.cpp bump would grow this repository by that much
# permanently, for everyone who clones it. The release workflow stages it into
# the tagged commit; locally you stage what you need.
#
set -euo pipefail
# Resolve arguments against the CALLER's directory, then move to our own. Without
# this, `stage.sh staged` from the repo root would look for natives/staged --
# which exists nowhere, and the error would blame the caller for the script's bug.
INVOKED_FROM="$PWD"
cd "$(dirname "$0")"
SRC="../core/dist"
REQUIRE_ALL=0
for a in "$@"; do
  # An `[ ... ] && VAR=1` here would be a set -e landmine: the loop's exit status
  # is the last test's, so a final argument that is not --require-all would end
  # the script with no output at all.
  if [ "$a" = "--require-all" ]; then
    REQUIRE_ALL=1
  else
    case "$a" in
      /*) SRC="$a" ;;
      *)  SRC="$INVOKED_FROM/$a" ;;
    esac
  fi
done

[ -d "$SRC" ] || { echo "no such directory: $SRC" >&2; exit 1; }

staged=0
for d in "$SRC"/*/; do
  key="$(basename "$d")"
  dest="payload/$key"
  [ -d "$dest" ] || { echo "  skip $key (not a platform this module carries)"; continue; }

  # A closure with no bridge in it is not a closure. Catch it here rather than at
  # dlopen, where the message is worse and the cause is further away.
  # One test per name: `ls a b` reports failure when EITHER is missing, so the
  # obvious one-liner rejects every platform.
  bridge=""
  for n in libllamabridge.dylib libllamabridge.so llamabridge.dll; do
    [ -e "$d/$n" ] && bridge="$n"
  done
  [ -n "$bridge" ] || { echo "  skip $key (no bridge library)"; continue; }
  [ -f "$d/links.json" ] || { echo "::error:: $key has no links.json — rebuild it" >&2; exit 1; }

  # -a to preserve the symlinks. They are dropped by go:embed regardless, which is
  # what links.json is for, but staging a faithful copy keeps this directory
  # comparable with core/dist/ and with what the other bindings ship.
  # Keep PLACEHOLDER: it is the only committed file here, and losing it makes
  # `git status` dirty after every local stage -- which trains people to ignore it.
  # Copied, not read into a variable: $(cat) strips trailing newlines, so writing
  # it back would leave the file "modified" in git for one invisible byte.
  keep="$(mktemp)"; kept=0
  if [ -f "$dest/PLACEHOLDER" ]; then cp "$dest/PLACEHOLDER" "$keep"; kept=1; fi
  rm -rf "$dest"; mkdir -p "$dest"
  cp -a "$d"/. "$dest"/
  [ "$kept" = 1 ] && cp "$keep" "$dest/PLACEHOLDER"
  rm -f "$keep"
  echo "  staged $key ($(du -sh "$dest" | cut -f1))"
  staged=$((staged+1))
done

[ "$staged" -gt 0 ] || { echo "nothing staged from $SRC" >&2; exit 1; }

if [ "$REQUIRE_ALL" = 1 ]; then
  missing=""
  for dest in payload/*/; do
    key="$(basename "$dest")"
    found=""
    for n in libllamabridge.dylib libllamabridge.so llamabridge.dll; do
      [ -e "$dest/$n" ] && found="$n"
    done
    [ -n "$found" ] || missing="$missing $key"
  done
  [ -z "$missing" ] || { echo "::error::no closure staged for:$missing" >&2; exit 1; }
  echo "==> every platform this module declares has a closure"
fi

echo "==> $staged platform(s) in payload/"
