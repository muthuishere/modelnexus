#!/usr/bin/env bash
#
# Run every example, in every language, and fail loudly on any error.
#
# bash, not sh. /bin/sh on Ubuntu is dash, which has no `pipefail` — a release in this
# repo has already been broken by that once, so the shebang is not decoration.
set -euo pipefail

# Models are large and are not in the repo or in CI. An example whose model is absent is
# SKIPPED, not failed — but a skipped run must never look like a passing one, so every
# skip is announced here and counted in the summary at the end.
#
#   MODELNEXUS_MODEL      a tool-capable chat GGUF        (hello, streaming, structured,
#                                                          conversation, counting, embeddings)
#   MODELNEXUS_RERANKER   a reranker GGUF, "rank" pooling (rerank)
#   MODELNEXUS_LORA_BASE  the base a LoRA was built for   (lora)
#   MODELNEXUS_LORA       the adapter itself              (lora)
#
# Run one language with: examples/run.sh go | python | js
# Run the integration examples with: examples/run.sh integrations

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/.." && pwd)

case "$(uname -s)" in Darwin) os=darwin ;; Linux) os=linux ;; *) os=unsupported ;; esac
case "$(uname -m)" in x86_64 | amd64) cpu=x86_64 ;; arm64 | aarch64) cpu=aarch64 ;; *) cpu=unknown ;; esac
platform="$os-$cpu"

# Every binding searches MODELNEXUS_LIB first. Setting it here means the examples run
# against the library this tree just built, not one left over in a package cache.
if [ -z "${MODELNEXUS_LIB:-}" ] && [ -d "$repo/core/dist/$platform" ]; then
  export MODELNEXUS_LIB="$repo/core/dist/$platform"
fi
if [ ! -d "${MODELNEXUS_LIB:-/nonexistent}" ]; then
  echo "no native bridge at core/dist/$platform — run: task build" >&2
  exit 1
fi

only="${1:-all}"

ran=0
skipped=0
skipped_names=()

have() { [ -n "${!1:-}" ] && [ -f "${!1}" ]; }

# run <language> <example> <required env var>... -- executes, or records a skip with the
# reason spelled out. A missing model is never silent.
run() {
  local lang=$1 name=$2
  shift 2
  local missing=()
  for var in "$@"; do
    have "$var" || missing+=("$var")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo
    echo "SKIP  $lang/$name — needs a model at ${missing[*]}"
    skipped=$((skipped + 1))
    skipped_names+=("$lang/$name (${missing[*]})")
    return 0
  fi

  echo
  echo "================================================================"
  echo "  $lang/$name"
  echo "================================================================"
  case "$lang" in
    go) (cd "$here/go" && CGO_ENABLED=0 go run "./$name") ;;
    python) (cd "$here/python" && PYTHONPATH="$repo/bindings/python" python3 "$name.py") ;;
    js) (cd "$here/js" && node "$name.js") ;;
  esac
  ran=$((ran + 1))
}

# The same eight examples in each language, in the same order, so two languages can be
# read side by side.
run_all() {
  local lang=$1
  run "$lang" hello MODELNEXUS_MODEL
  run "$lang" streaming MODELNEXUS_MODEL
  run "$lang" structured MODELNEXUS_MODEL
  run "$lang" conversation MODELNEXUS_MODEL
  run "$lang" counting MODELNEXUS_MODEL
  run "$lang" embeddings MODELNEXUS_MODEL
  run "$lang" rerank MODELNEXUS_RERANKER
  run "$lang" lora MODELNEXUS_LORA_BASE MODELNEXUS_LORA
}

# ---------------------------------------------------------------- integrations
#
# modelnexus behind somebody else's library. These are SEPARATE from the list above
# for two reasons that both matter:
#
#   1. Each is its own Go module, because it pulls a dependency the other examples
#      have no business acquiring. `go run ./name` from examples/go cannot reach them.
#   2. They prove a different claim. The eight above show what modelnexus does; these
#      show that a general-purpose library can drive it with no knowledge of it, and
#      that neither side depends on the other (ADR-0003).
#
# They are run against the PUBLISHED integration library, never a local checkout --
# otherwise they prove something about a working tree rather than about a release.
run_module() {
  local name=$1 desc=$2
  shift 2
  local missing=()
  for var in "$@"; do have "$var" || missing+=("$var"); done
  if [ ${#missing[@]} -gt 0 ]; then
    echo
    echo "SKIP  integrations/$name — needs a model at ${missing[*]}"
    skipped=$((skipped + 1))
    skipped_names+=("integrations/$name (${missing[*]})")
    return 0
  fi
  echo
  echo "================================================================"
  echo "  integrations/$name — $desc"
  echo "================================================================"
  (cd "$here/go/$name" && CGO_ENABLED=0 go run .)
  ran=$((ran + 1))
}

run_integrations() {
  run_module toolnexus "an agent with tools, model in this process" MODELNEXUS_MODEL
  run_module citenexus "embeddings and grounded answers" MODELNEXUS_MODEL
}

if [ "$only" = "all" ] || [ "$only" = "js" ]; then
  # The examples depend on the binding in this tree through a file: link, so the install
  # is local and offline. Do it once, before any JS example runs.
  (cd "$here/js" && npm install --silent --no-audit --no-fund)
fi

case "$only" in
  all) run_all go; run_all python; run_all js; run_integrations ;;
  go | python | js) run_all "$only" ;;
  integrations) run_integrations ;;
  *) echo "usage: run.sh [all|go|python|js|integrations]" >&2; exit 2 ;;
esac

echo
echo "================================================================"
echo "  $ran example(s) ran, $skipped skipped"
if [ "$skipped" -gt 0 ]; then
  # Loud on purpose. A run where every model was absent exits 0, and this block is the
  # only thing standing between that and a green tick that means nothing.
  echo
  echo "  SKIPPED — no model on this machine:"
  for s in "${skipped_names[@]}"; do echo "    - $s"; done
  echo
  echo "  These examples were NOT executed. Point MODELNEXUS_MODEL / MODELNEXUS_RERANKER /"
  echo "  MODELNEXUS_LORA_BASE + MODELNEXUS_LORA at real GGUF files to run them."
fi
if [ "$ran" -eq 0 ]; then
  echo
  echo "  NOTHING RAN. This run proves nothing about the examples."
fi
echo "================================================================"
