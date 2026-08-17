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

# The JS probes must live INSIDE the package for a relative import to resolve,
# so they cannot go in $OUT. They are therefore PID-suffixed: fixed names meant
# two concurrent runs deleted each other's probe through the EXIT trap, and a
# perfectly correct binding reported MODULE_NOT_FOUND.
JS_AGREE="$ROOT/bindings/js/agree.$$.tmp.mjs"
JS_CFG="$ROOT/bindings/js/cfg.$$.tmp.mjs"

trap 'rm -rf "$OUT"; rm -f "$JS_AGREE" "$JS_CFG"' EXIT

# One prompt, one schema, temperature 0, fixed seed. Anything that differs
# between bindings after this is the binding's doing.
PROMPT='Describe Paris.'
SCHEMA='{"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"}},"required":["city","country"],"additionalProperties":false}'
SEED=42

echo "model:  $MODELNEXUS_MODEL"
echo "prompt: $PROMPT"
echo

broken=""

run() {   # run <name> <command...>
  local name="$1"; shift
  if "$@" > "$OUT/$name" 2> "$OUT/$name.err"; then
    printf '  %-10s %s\n' "$name" "$(cat "$OUT/$name")"
  else
    printf '  %-10s FAILED (see below)\n' "$name"
    sed 's/^/             /' "$OUT/$name.err" | tail -5
    rm -f "$OUT/$name"
    # A probe that fails is NOT a probe that is absent. Dropping it silently
    # turned a broken binding into one that simply did not participate, and the
    # summary below cheerfully said AGREE about the survivors.
    broken="$broken $name"
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
from modelnexus import Chat, Embedder
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
cp "$OUT/agree.mjs" "$JS_AGREE"

echo "answers (must be byte-identical):"
run go     bash -c "cd '$ROOT/bindings/go' && go run '$OUT/agree.go' '$MODELNEXUS_MODEL' '$SCHEMA' '$PROMPT'"
run python bash -c "cd '$ROOT/bindings/python' && PYTHONPATH='$ROOT/bindings/python' python3 '$OUT/agree.py' '$MODELNEXUS_MODEL' '$SCHEMA' '$PROMPT'"
run js     bash -c "cd '$ROOT/bindings/js' && node '$JS_AGREE' '$MODELNEXUS_MODEL' '$SCHEMA' '$PROMPT'"

echo
# grep -E, not -v with \|: BSD grep does not support alternation in a BASIC
# regex, so on macOS the plain form silently matches nothing and the .err
# files get compared as though they were answers.
# ======================================================================
# Phase 2 - CREATION.
#
# Comparing inference output proves the bindings agree on what the core
# RETURNS. It says nothing about what they SEND, and two bindings sending
# different configs can behave identically today and diverge the moment a core
# default moves. So: each binding creates engines for three intents and reports
# the config the CORE actually received, through the event callback every
# binding already exposes. Observed, not self-reported.
# ======================================================================

cat > "$OUT/cfg.go" <<'GOEOF'
package main

import (
	"fmt"
	"os"
	"strings"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

func probe(model, label string, opts ...modelnexus.Option) {
	var seen string
	all := append([]modelnexus.Option{modelnexus.WithEventHandler(func(e string) {
		if strings.HasPrefix(e, "create_config:") {
			seen = strings.TrimPrefix(e, "create_config:")
		}
	})}, opts...)
	c, err := modelnexus.Open(model, all...)
	if err != nil {
		fmt.Fprintln(os.Stderr, label, err)
		os.Exit(1)
	}
	defer c.Close()
	fmt.Printf("%s=%s\n", label, seen)
}

func main() {
	m := os.Args[1]
	probe(m, "chat-none")
	probe(m, "chat-ctx", modelnexus.WithContextSize(2048))
	probe(m, "chat-full", modelnexus.WithContextSize(2048), modelnexus.WithBatchSize(256), modelnexus.WithMaxSequences(1))
	// 0 is CPU-only, a legitimate value -- and the one a zero-means-unset binding
	// silently drops, sending nothing and getting the GPU instead.
	probe(m, "chat-gpu0", modelnexus.WithGPULayers(0))
	probeEmbed(m, "embed-none", nil)
	probeEmbed(m, "embed-pool", &modelnexus.EmbedOptions{Pooling: modelnexus.PoolingMean})
	zero := 0
	probeEmbed(m, "embed-gpu0", &modelnexus.EmbedOptions{NGPULayers: &zero})
}

func probeEmbed(model, label string, opts *modelnexus.EmbedOptions) {
	var seen string
	o := opts
	if o == nil {
		o = &modelnexus.EmbedOptions{}
	}
	o.OnEvent = func(e string) {
		if strings.HasPrefix(e, "create_config:") {
			seen = strings.TrimPrefix(e, "create_config:")
		}
	}
	e, err := modelnexus.OpenEmbedder(model, o)
	if err != nil {
		fmt.Fprintln(os.Stderr, label, err)
		os.Exit(1)
	}
	defer e.Close()
	fmt.Printf("%s=%s\n", label, seen)
}
GOEOF

cat > "$OUT/cfg.py" <<'PYEOF'
import sys
from modelnexus import Chat, Embedder

def probe(model, label, **kw):
    seen = []
    def on_event(e):
        if e.startswith("create_config:"):
            seen.append(e[len("create_config:"):])
    with Chat(model, on_event=on_event, **kw):
        pass
    print(f"{label}={seen[-1] if seen else '<none>'}")

m = sys.argv[1]
probe(m, "chat-none")
probe(m, "chat-ctx", n_ctx=2048)
probe(m, "chat-full", n_ctx=2048, n_batch=256, n_seq_max=1)
probe(m, "chat-gpu0", n_gpu_layers=0)

def probe_embed(model, label, **kw):
    seen = []
    def on_event(e):
        if e.startswith("create_config:"):
            seen.append(e[len("create_config:"):])
    e = Embedder(model, on_event=on_event, **kw)
    try:
        print(f"{label}={seen[-1] if seen else '<none>'}")
    finally:
        e.close()

probe_embed(m, "embed-none")
probe_embed(m, "embed-pool", pooling="mean")
probe_embed(m, "embed-gpu0", n_gpu_layers=0)
PYEOF

cat > "$OUT/cfg.mjs" <<'JSEOF'
import { Chat, Embedder } from './src/index.js'
const m = process.argv[2]
function probe(label, options = {}) {
  let seen = '<none>'
  const c = new Chat(m, { ...options, onEvent: (e) => {
    if (e.startsWith('create_config:')) seen = e.slice('create_config:'.length)
  } })
  try { console.log(`${label}=${seen}`) } finally { c.close() }
}
probe('chat-none')
probe('chat-ctx', { nCtx: 2048 })
probe('chat-full', { nCtx: 2048, nBatch: 256, nSeqMax: 1 })
probe('chat-gpu0', { nGpuLayers: 0 })

function probeEmbed(label, options = {}) {
  let seen = '<none>'
  const e = new Embedder(m, { ...options, onEvent: (ev) => {
    if (ev.startsWith('create_config:')) seen = ev.slice('create_config:'.length)
  } })
  try { console.log(`${label}=${seen}`) } finally { e.close() }
}
probeEmbed('embed-none')
probeEmbed('embed-pool', { pooling: 'mean' })
probeEmbed('embed-gpu0', { nGpuLayers: 0 })
JSEOF

cp "$OUT/cfg.mjs" "$JS_CFG"

echo
echo "create configs (must be byte-identical):"
run go-cfg     bash -c "cd '$ROOT/bindings/go' && go run '$OUT/cfg.go' '$MODELNEXUS_MODEL'"
run python-cfg bash -c "cd '$ROOT/bindings/python' && PYTHONPATH='$ROOT/bindings/python' python3 '$OUT/cfg.py' '$MODELNEXUS_MODEL'"
run js-cfg     bash -c "cd '$ROOT/bindings/js' && node '$JS_CFG' '$MODELNEXUS_MODEL'"
# C# is not published (ADR-0007) but must stay at parity — and it was the ONE
# binding this gate could not see, which is exactly where an Embedder that
# always sent {"n_batch":512} survived unnoticed. It cannot join the inference
# phase (that needs a script, and this needs a compiled project), but creation
# it can answer, and creation is where the divergence was.
run csharp-cfg bash -c "cd '$ROOT/bindings/csharp' && dotnet run --project Smoke -- --create-config '$MODELNEXUS_MODEL' 2>/dev/null"

# Canonicalise before comparing. json.dumps puts spaces after ':' by default and
# Go's encoding/json sorts map keys -- both cosmetic for a config object, and
# both would drown the real differences. `{}` vs `null` survives canonicalisation
# because it IS a real difference: one sends a config, the other sends none.
for f in "$OUT"/*-cfg; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'CANONEOF' > "$f.canon" && mv "$f.canon" "$f"
import json, sys
for line in open(sys.argv[1]):
    line = line.rstrip("\n")
    if not line: continue
    label, _, raw = line.partition("=")
    try:
        raw = json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"))
    except Exception:
        pass
    print(f"{label}={raw}")
CANONEOF
done

cfgs=$(ls "$OUT" | grep -E -- '-cfg$' | sort)
cfgcount=$(echo "$cfgs" | grep -c . || true)
cfgfirst=""; cfgdisagree=0
for f in $cfgs; do
  body=$(cat "$OUT/$f")
  if [ -z "$cfgfirst" ]; then cfgfirst="$body"; continue; fi
  [ "$body" = "$cfgfirst" ] || cfgdisagree=1
done

present=$(ls "$OUT" | grep -vE '\.err$|^agree\.|^cfg\.|-cfg$' | sort)
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

fail=0
if [ -n "$broken" ]; then
  echo "FAILED to run:$broken — a binding that cannot answer is not a binding that agrees." >&2
  fail=1
fi
if [ "$disagree" -ne 0 ]; then
  echo "DISAGREE (output) — the bindings are not returning the same thing:" >&2
  for f in $present; do printf '  %-10s %s\n' "$f" "$(cat "$OUT/$f")" >&2; done
  fail=1
fi
if [ "$cfgcount" -lt 2 ]; then
  echo "INCONCLUSIVE (create config): fewer than two bindings reported" >&2
  fail=1
elif [ "$cfgdisagree" -ne 0 ]; then
  echo "DISAGREE (create config) — same intent, different payload:" >&2
  for f in $cfgs; do
    printf '  %s\n' "$f" >&2
    sed 's/^/      /' "$OUT/$f" >&2
  done
  echo "  A binding that restates a default the core owns pins the old value" >&2
  echo "  the day the core changes it. Send only what the caller set." >&2
  fail=1
fi
[ "$fail" -eq 0 ] || exit 1
echo "AGREE — identical text and token count across $count bindings;" \
     "identical create config across $cfgcount."
exit 0
