#!/usr/bin/env bash
#
# Copy staged native closures into the natives module's payload directories.
#
#   ./stage.sh                    from core/dist/ (whatever you have built locally)
#   ./stage.sh path/to/staged     from a directory of <platform-key>/ trees
#
# The payload is NOT committed. Five platforms at ~14 MB each is ~70 MB, and a
# fresh copy on every llama.cpp bump would grow this repository by that much
# permanently, for everyone who clones it. The release workflow stages it into
# the tagged commit; locally you stage what you need.
#
set -euo pipefail
cd "$(dirname "$0")"
SRC="${1:-../core/dist}"

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
  rm -rf "$dest"; mkdir -p "$dest"
  cp -a "$d"/. "$dest"/
  echo "  staged $key ($(du -sh "$dest" | cut -f1))"
  staged=$((staged+1))
done

[ "$staged" -gt 0 ] || { echo "nothing staged from $SRC" >&2; exit 1; }
echo "==> $staged platform(s) in payload/"
