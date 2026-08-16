#!/usr/bin/env bash
# Build and run spike 0003. Requires core/build.sh to have run once (it leaves
# the headers checkout and the prebuilt llama.cpp libs where this looks).
#
#   ./run.sh                 # all five questions, default model
#   ./run.sh prefix          # one question
#   ./run.sh all <model>     # explicit model
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WHAT="${1:-all}"
MODEL="${2:-$HOME/.chatbot_models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"

HEADERS="$ROOT/core/third_party/llama.cpp"
PREBUILT="$(dirname "$(find "$ROOT/core/build/llama-prebuilt" -name 'libllama.dylib' -o -name 'libllama.so' 2>/dev/null | head -1)")"

[ -f "$HEADERS/include/llama.h" ] || { echo "no headers at $HEADERS — run core/build.sh first" >&2; exit 1; }
[ -d "$PREBUILT" ] || { echo "no prebuilt llama.cpp — run core/build.sh first" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }

cmake -S "$HERE" -B "$HERE/build" \
  -DLLB_HEADERS_DIR="$HEADERS" -DLLB_PREBUILT_DIR="$PREBUILT" >/dev/null
cmake --build "$HERE/build" --config Release -j >/dev/null

exec "$HERE/build/spike" "$WHAT" "$MODEL"
