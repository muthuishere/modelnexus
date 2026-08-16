#!/usr/bin/env bash
# Cross-binding agreement — the parity gate (ADR-0002).
#
# Every binding runs the SAME schema-constrained request and the SAME token
# count against the SAME model at temperature 0, and must produce byte-identical
# answers. Per-binding suites prove each binding works; this proves they agree,
# which is a different claim and the one the ABI actually makes.
#
# Deliberately compares OUTPUT, not internals: a binding that quietly reshapes a
# result would pass its own tests and fail here.
#
#   ./core/tests/agreement.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
: "${MODELNEXUS_MODEL:=$HOME/.chatbot_models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
export MODELNEXUS_MODEL

if [ ! -f "$MODELNEXUS_MODEL" ]; then
  echo "SKIP: no model at $MODELNEXUS_MODEL — set MODELNEXUS_MODEL" >&2
  exit 0
fi

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# One prompt, one schema, temperature 0, fixed seed. Anything that differs
# between bindings after this is the binding's doing.
PROMPT='Describe Paris.'
SCHEMA='{"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"}},"required":["city","country"],"additionalProperties":false}'
SEED=42

echo "model:  $MODELNEXUS_MODEL"
echo "prompt: $PROMPT"
echo

run() {   # run <name> <command...>
  local name="$1"; shift
  if "$@" > "$OUT/$name" 2> "$OUT/$name.err"; then
    printf '  %-8s %s\n' "$name" "$(cat "$OUT/$name")"
  else
    printf '  %-8s FAILED (see below)\n' "$name"
    sed 's/^/           /' "$OUT/$name.err" | tail -5
    rm -f "$OUT/$name"
  fi
}

# --- Go ----------------------------------------------------------------
cat > "$OUT/agree.go" <<'GO'
package main

import (
	"encoding/json"
	"fmt"
	"os"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

func main() {
	var schema any
	_ = json.Unmarshal([]byte(os.Args[2]), &schema)
	c, err := modelnexus.Open(os.Args[1])
	if err != nil { fmt.Fprintln(os.Stderr, err); os.Exit(1) }
	defer c.Close()
	// Every generation parameter is a POINTER in this binding, deliberately:
	// a Temperature of 0 is a legitimate request and must be distinguishable
	// from "unset". That is why these are addresses of locals.
	maxTokens := 120
	var seed uint32 = 42
	temp := 0.0
	req := modelnexus.Request{
		Messages:    []modelnexus.Message{{Role: "user", Content: os.Args[3]}},
		MaxTokens:   &maxTokens,
		Seed:        &seed,
		Temperature: &temp,
		JSONSchema:  schema,
	}
	resp, err := c.Infer(req)
	if err != nil { fmt.Fprintln(os.Stderr, err); os.Exit(1) }
	tc, err := c.CountTokens(req)
	if err != nil { fmt.Fprintln(os.Stderr, err); os.Exit(1) }
	fmt.Printf("%s | tokens=%d\n", resp.Text, tc.Tokens)
}
GO

# --- Python ------------------------------------------------------------
cat > "$OUT/agree.py" <<'PY'
import json, sys
from modelnexus import Chat
schema = json.loads(sys.argv[2])
with Chat(sys.argv[1]) as c:
    msgs = [{"role": "user", "content": sys.argv[3]}]
    r = c.infer(msgs, max_tokens=120, seed=42, temperature=0.0, json_schema=schema)
    n = c.count_tokens(msgs)
    print(f"{r['text']} | tokens={n['tokens']}")
PY

# --- JS ----------------------------------------------------------------
cat > "$OUT/agree.mjs" <<'JS'
import { Chat } from './src/index.js'
const [model, schemaRaw, prompt] = process.argv.slice(2)
const c = new Chat(model)
try {
  const req = {
    messages: [{ role: 'user', content: prompt }],
    maxTokens: 120, seed: 42, temperature: 0,
    jsonSchema: JSON.parse(schemaRaw),
  }
  const r = c.infer(req)
  const n = c.countTokens({ messages: req.messages })
  console.log(`${r.text} | tokens=${n.tokens}`)
} finally { c.close() }
JS

# The JS script must live inside the package so a bare relative import
# resolves; a temp dir cannot see node_modules or the package entry point.
cp "$OUT/agree.mjs" "$ROOT/bindings/js/agree.tmp.mjs"
trap 'rm -rf "$OUT"; rm -f "$ROOT/bindings/js/agree.tmp.mjs"' EXIT

echo "answers (must be byte-identical):"
run go     bash -c "cd '$ROOT/bindings/go' && go run '$OUT/agree.go' '$MODELNEXUS_MODEL' '$SCHEMA' '$PROMPT'"
run python bash -c "cd '$ROOT/bindings/python' && PYTHONPATH='$ROOT/bindings/python' python3 '$OUT/agree.py' '$MODELNEXUS_MODEL' '$SCHEMA' '$PROMPT'"
run js     bash -c "cd '$ROOT/bindings/js' && node '$ROOT/bindings/js/agree.tmp.mjs' '$MODELNEXUS_MODEL' '$SCHEMA' '$PROMPT'"

echo
# grep -E, not -v with \|: BSD grep does not support alternation in a BASIC
# regex, so on macOS the plain form silently matches nothing and the .err
# files get compared as though they were answers.
present=$(ls "$OUT" | grep -vE '\.err$|^agree\.' | sort)
count=$(echo "$present" | grep -c . || true)
if [ "$count" -lt 2 ]; then
  echo "INCONCLUSIVE: fewer than two bindings produced an answer" >&2
  exit 1
fi

first=""; disagree=0
for f in $present; do
  body=$(cat "$OUT/$f")
  if [ -z "$first" ]; then first="$body"; continue; fi
  [ "$body" = "$first" ] || disagree=1
done

if [ "$disagree" -eq 0 ]; then
  echo "AGREE — $count bindings, identical text and identical token count."
  exit 0
fi
echo "DISAGREE — the bindings are not returning the same thing:" >&2
for f in $present; do printf '  %-8s %s\n' "$f" "$(cat "$OUT/$f")" >&2; done
exit 1
