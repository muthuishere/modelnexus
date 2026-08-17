#!/usr/bin/env bash
#
# Build the modelnexus native bridge.
#
# This is the orchestration mochallama does from Gradle, in plain bash — because a
# Go or Python consumer has no reason to install a JVM toolchain to get a shared
# library. The CMakeLists it drives is mochallama's, unchanged in substance.
#
#   ./build.sh                 prebuilt mode (default) — download llama.cpp's official
#                              release libs, compile ONLY our one-file bridge. Seconds.
#   ./build.sh --source        source mode — build llama.cpp from the checkout too.
#                              Slow (tens of minutes), but it is yours end to end and
#                              it is the fallback when an upstream asset is broken.
#   ./build.sh --clean         remove build/ and dist/ first.
#   ./build.sh --platform KEY  cross-build for another platform key. Only
#                              darwin-x86_64 from an arm64 Mac is supported, and only
#                              because prebuilt mode never compiles llama.cpp — this
#                              is one clang invocation, not a port (ADR-0010).
#
# Output: dist/<platform-key>/ containing libllamabridge plus the llama/ggml libs it
# needs at runtime. The bridge resolves its siblings via rpath @loader_path/$ORIGIN,
# so that directory is self-contained — hand it to any FFI binding as-is.
#
set -euo pipefail

# Pinned llama.cpp release. Single source of truth for BOTH the prebuilt binaries we
# download and the headers we compile against. Bump deliberately: it is part of the
# release identity (ADR-0004).
LLAMA_TAG="${LLAMA_TAG:-b9371}"

cd "$(dirname "$0")"
ROOT="$PWD"
MODE="prebuilt"
TARGET=""
want_platform=0
for arg in "$@"; do
  if [ "$want_platform" = 1 ]; then TARGET="$arg"; want_platform=0; continue; fi
  case "$arg" in
    --source) MODE="source" ;;
    --clean)  rm -rf "$ROOT/build" "$ROOT/dist" ;;
    --platform) want_platform=1 ;;
    --platform=*) TARGET="${arg#--platform=}" ;;
    # One place knows how to read the pin. Anything else parsing this file with sed
    # is a second source of truth waiting to drift.
    --print-tag) echo "$LLAMA_TAG"; exit 0 ;;
    --help|-h) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $arg (try --help)" >&2; exit 2 ;;
  esac
done
[ "$want_platform" = 0 ] || { echo "--platform needs a key, e.g. darwin-x86_64" >&2; exit 2; }

# ---- platform key -----------------------------------------------------------
case "$(uname -s)" in
  Darwin) OS=darwin; EXT=.dylib ;;
  Linux)  OS=linux;  EXT=.so ;;
  *) echo "unsupported OS: $(uname -s). Windows builds run in CI." >&2; exit 1 ;;
esac
case "$(uname -m)" in
  x86_64|amd64)  ARCH=x86_64 ;;
  arm64|aarch64) ARCH=aarch64 ;;
  *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;;
esac
PLATFORM="$OS-$ARCH"

# ---- cross-build ------------------------------------------------------------
# Only one cross is supported, and only because prebuilt mode never compiles
# llama.cpp: upstream already publishes the x86_64 dylibs, so this compiles our
# ~890-line bridge for another arch and stages upstream's libraries beside it.
# That is why no Intel runner is needed (ADR-0010, spike 0009).
#
# It is NOT a general cross-compilation facility. A key this build cannot honestly
# produce is refused rather than silently mislabelled -- a dist/ named for a
# platform it cannot run on is worse than no dist/ at all.
CROSS_ARGS=()
if [ -n "$TARGET" ] && [ "$TARGET" != "$PLATFORM" ]; then
  case "$PLATFORM/$TARGET" in
    darwin-aarch64/darwin-x86_64|darwin-x86_64/darwin-aarch64)
      CROSS_ARGS+=("-DCMAKE_OSX_ARCHITECTURES=${TARGET#darwin-}")
      # CMake spells Apple arm64 "arm64", our key spells it "aarch64".
      [ "$TARGET" = "darwin-aarch64" ] && CROSS_ARGS=("-DCMAKE_OSX_ARCHITECTURES=arm64")
      echo "==> cross-building $TARGET on $PLATFORM"
      PLATFORM="$TARGET"
      ;;
    *)
      echo "cannot cross-build $TARGET from $PLATFORM" >&2
      echo "  supported: darwin-x86_64 <-> darwin-aarch64 (Apple ships the cross toolchain)" >&2
      exit 2
      ;;
  esac
fi

# llama.cpp names its release assets differently from our platform key.
case "$PLATFORM" in
  darwin-x86_64)  ASSET=macos-x64 ;;
  darwin-aarch64) ASSET=macos-arm64 ;;
  linux-x86_64)   ASSET=ubuntu-x64 ;;
  linux-aarch64)  ASSET=ubuntu-arm64 ;;
  *) echo "no prebuilt llama.cpp asset mapping for $PLATFORM" >&2; exit 1 ;;
esac

HEADERS="$ROOT/third_party/llama.cpp"
# Per-platform: a cross-build must not reuse the host's CMake cache, which pins the
# architecture from the first configure and would silently produce the wrong one.
BUILD="$ROOT/build/cmake/$PLATFORM"
DIST="$ROOT/dist/$PLATFORM"
PREBUILT="$ROOT/build/llama-prebuilt"

echo "==> modelnexus native bridge | platform=$PLATFORM mode=$MODE llama.cpp=$LLAMA_TAG"

# ---- headers ----------------------------------------------------------------
# Needed in BOTH modes for include paths. In prebuilt mode this tree is never
# compiled, only read. It is gitignored: cloned on demand, never vendored, so the
# repo stays small and the tag stays the single source of truth.
if [ ! -f "$HEADERS/include/llama.h" ]; then
  echo "==> cloning llama.cpp $LLAMA_TAG (shallow)"
  rm -rf "$HEADERS"
  git clone --depth 1 -b "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$HEADERS"
fi

mkdir -p "$BUILD" "$DIST"
CMAKE_ARGS=(-S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "-DLLB_HEADERS_DIR=$HEADERS" "${CROSS_ARGS[@]+"${CROSS_ARGS[@]}"}")

if [ "$MODE" = "prebuilt" ]; then
  ARCHIVE="llama-$LLAMA_TAG-bin-$ASSET.tar.gz"
  URL="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/$ARCHIVE"
  EXTRACT="$PREBUILT/$PLATFORM"
  if [ ! -d "$EXTRACT" ]; then
    echo "==> downloading $URL"
    mkdir -p "$EXTRACT"
    curl -fsSL "$URL" -o "$PREBUILT/$ARCHIVE"
    tar -xzf "$PREBUILT/$ARCHIVE" -C "$EXTRACT"
  fi
  # find_library needs the exact directory holding libllama.*
  LIBDIR="$(dirname "$(find "$EXTRACT" -name "libllama$EXT" -o -name "libllama$EXT.*" | head -1)")"
  [ -n "$LIBDIR" ] && [ -d "$LIBDIR" ] || { echo "prebuilt archive had no llama shared lib" >&2; exit 1; }
  echo "==> linking against prebuilt libs in $LIBDIR"
  CMAKE_ARGS+=("-DLLB_PREBUILT_DIR=$LIBDIR")
  LIB_SOURCE="$LIBDIR"
else
  echo "==> building llama.cpp from source (this takes a while)"
  LIB_SOURCE=""   # resolved after the build
fi

# ---- build ------------------------------------------------------------------
cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD" --config Release -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

# ---- stage ------------------------------------------------------------------
# The bridge plus every llama/ggml shared library it loads at runtime, side by side.
BRIDGE="$(find "$BUILD" -name "libllamabridge$EXT" | head -1)"
[ -n "$BRIDGE" ] || { echo "bridge not produced" >&2; exit 1; }
cp "$BRIDGE" "$DIST/"

if [ "$MODE" = "source" ]; then
  LIB_SOURCE="$BUILD"
fi
# The bridge links @rpath/libllama.0.dylib and @rpath/libllama-common.0.dylib —
# VERSIONED names, which a glob like "libllama.dylib*" silently misses, producing a
# dist/ that looks complete and fails to load. So match the versioned forms too.
#
# But not everything: llama.cpp's release archives also ship libllama-*-impl libraries
# (cli, server, bench, quantize). Those are tool implementations, not runtime deps of
# the bridge, and shipping them would roughly double dist/ for nothing.
# -a, not -f: llama.cpp ships libfoo.dylib -> libfoo.0.dylib -> libfoo.0.0.N.dylib as
# SYMLINKS. Dereferencing them turns one 7.5 MB library into three, which is most of
# the weight of a naive dist/ and is pure duplication.
find "$LIB_SOURCE" -maxdepth 2 \( -name "*$EXT" -o -name "*$EXT.*" -o -name "*.[0-9]*$EXT" \) \
  ! -name "*-impl*" -exec cp -a {} "$DIST/" \; 2>/dev/null || true

# Every notice in licenses/, not just llama.cpp's. nlohmann/json is header-only and
# compiled INTO the bridge, so its MIT notice has to travel with the binary too.
cp "$ROOT"/licenses/* "$DIST/" 2>/dev/null || true

# ---- link manifest ----------------------------------------------------------
# The symlinks above are load-bearing: the bridge links @rpath/libllama.0.dylib,
# which IS one. But some distribution mechanisms cannot carry a symlink -- go:embed
# takes regular files only, records nothing, and reports nothing, so a naive bundle
# yields a closure that looks complete and cannot load (spike 0009).
#
# So the build records them. Generated from what was actually staged, which is the
# only way it cannot drift from what ships. Empty on Windows; same shape everywhere,
# because a consumer branching on platform is a consumer with a bug waiting.
python3 - "$DIST" <<'PY'
import json, os, sys
dist = sys.argv[1]
links = {}
for name in sorted(os.listdir(dist)):
    p = os.path.join(dist, name)
    if os.path.islink(p):
        links[name] = os.readlink(p)

# A link whose target is missing is a closure that will fail at dlopen, far from
# here and with a worse message. Refuse to stage it.
broken = [f"{n} -> {t}" for n, t in links.items()
          if not os.path.exists(os.path.join(dist, t))]
if broken:
    sys.exit("dangling symlink(s) in the staged closure:\n  " + "\n  ".join(broken))

with open(os.path.join(dist, "links.json"), "w") as f:
    json.dump({"links": links}, f, indent=1, sort_keys=True)
print(f"==> link manifest: {len(links)} symlink(s)")
PY

echo "==> staged to $DIST"
ls -1 "$DIST"
