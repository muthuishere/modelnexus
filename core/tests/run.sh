#!/usr/bin/env bash
# Build and run the C-level ABI conformance test against the staged bridge.
# No language runtime involved — that is the point.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DIST="$(ls -d "$ROOT"/core/dist/*/ 2>/dev/null | head -1)"
[ -d "$DIST" ] || { echo "no core/dist — run core/build.sh first" >&2; exit 1; }

: "${MODELNEXUS_MODEL:=$HOME/.chatbot_models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
export MODELNEXUS_MODEL

cc -std=c11 -O1 -I"$ROOT/core/include" "$HERE/abi_test.c" \
   -L"$DIST" -lllamabridge -Wl,-rpath,"$DIST" -o "$HERE/abi_test"
exec "$HERE/abi_test"
