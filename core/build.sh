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
for arg in "$@"; do
  case "$arg" in
    --source) MODE="source" ;;
    --clean)  rm -rf "$ROOT/build" "$ROOT/dist" ;;
    # One place knows how to read the pin. Anything else parsing this file with sed
    # is a second source of truth waiting to drift.
    --print-tag) echo "$LLAMA_TAG"; exit 0 ;;
    --help|-h) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $arg (try --help)" >&2; exit 2 ;;
  esac
done

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

# llama.cpp names its release assets differently from our platform key.
case "$PLATFORM" in
  darwin-x86_64)  ASSET=macos-x64 ;;
  darwin-aarch64) ASSET=macos-arm64 ;;
  linux-x86_64)   ASSET=ubuntu-x64 ;;
  linux-aarch64)  ASSET=ubuntu-arm64 ;;
  *) echo "no prebuilt llama.cpp asset mapping for $PLATFORM" >&2; exit 1 ;;
esac

HEADERS="$ROOT/third_party/llama.cpp"
BUILD="$ROOT/build/cmake"
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
CMAKE_ARGS=(-S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "-DLLB_HEADERS_DIR=$HEADERS")

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

echo "==> staged to $DIST"
ls -1 "$DIST"
