#!/usr/bin/env bash
#
# SPIKE 0009 / question 1: can an Apple Silicon runner produce a WORKING
# darwin-x86_64 native, so no Intel macOS runner is needed at all?
#
# This replicates core/build.sh's prebuilt path with one addition --
# CMAKE_OSX_ARCHITECTURES=x86_64 -- and stages to a spike directory. Throwaway on
# purpose: if the verdict is yes, the change lands in build.sh as a flag, not as
# a copy of this file.
#
set -euo pipefail
cd "$(dirname "$0")"
SPIKE="$PWD"
ROOT="$(cd ../../core && pwd)"

LLAMA_TAG="$(bash "$ROOT/build.sh" --print-tag)"
PLATFORM=darwin-x86_64
ASSET=macos-x64

HEADERS="$ROOT/third_party/llama.cpp"
BUILD="$SPIKE/build"
DIST="$SPIKE/dist/$PLATFORM"
PREBUILT="$SPIKE/llama-prebuilt"

echo "==> cross-building $PLATFORM on $(uname -m) | llama.cpp=$LLAMA_TAG"

# llama.cpp already PUBLISHES the x86_64 dylibs. That is the whole reason this is
# cheap: nothing of llama.cpp is compiled here, only our one-file bridge.
ARCHIVE="llama-$LLAMA_TAG-bin-$ASSET.tar.gz"
EXTRACT="$PREBUILT/$PLATFORM"
if [ ! -d "$EXTRACT" ]; then
  mkdir -p "$EXTRACT"
  curl -fsSL "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/$ARCHIVE" \
    -o "$PREBUILT/$ARCHIVE"
  tar -xzf "$PREBUILT/$ARCHIVE" -C "$EXTRACT"
fi
LIBDIR="$(dirname "$(find "$EXTRACT" -name "libllama.dylib" -o -name "libllama.dylib.*" | head -1)")"
[ -d "$LIBDIR" ] || { echo "no prebuilt llama lib"; exit 1; }

# Confirm the DOWNLOADED libraries really are x86_64 before blaming our build for
# anything downstream.
echo "==> upstream libllama arch: $(lipo -archs "$LIBDIR/libllama.dylib" 2>/dev/null || file -b "$LIBDIR/libllama.dylib")"

mkdir -p "$BUILD" "$DIST"
cmake -S "$ROOT" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DLLB_HEADERS_DIR="$HEADERS" \
  -DLLB_PREBUILT_DIR="$LIBDIR" >/dev/null
cmake --build "$BUILD" --config Release -j "$(getconf _NPROCESSORS_ONLN)" >/dev/null

BRIDGE="$(find "$BUILD" -name "libllamabridge.dylib" | head -1)"
[ -n "$BRIDGE" ] || { echo "bridge not produced"; exit 1; }
cp "$BRIDGE" "$DIST/"
find "$LIBDIR" -maxdepth 2 \( -name "*.dylib" -o -name "*.dylib.*" -o -name "*.[0-9]*.dylib" \) \
  ! -name "*-impl*" -exec cp -a {} "$DIST/" \; 2>/dev/null || true

echo "==> staged $(ls -1 "$DIST" | wc -l | tr -d ' ') files to $DIST"
echo "==> bridge arch: $(lipo -archs "$DIST/libllamabridge.dylib")"
lipo -archs "$DIST/libllamabridge.dylib" | grep -q x86_64 || { echo "FAIL: not x86_64"; exit 1; }
echo "==> compiles: YES"
