"""Does the PUBLISHED Windows native actually load and run?

Driven by verify-windows.cmd. Deliberately exercises the 0.2.0-only entry
points, so it CANNOT pass against the 0.1.0 bridge -- the same trick that
caught the stale natives on macOS.

Exits non-zero on the first real failure. A missing model is a SKIP of the
inference half, reported loudly, never a silent pass.
"""

from __future__ import annotations

import os
import sys

checks = 0
failures = 0


def ok(cond: bool, what: str, detail: str = "") -> None:
    global checks, failures
    checks += 1
    if cond:
        print(f"  ok    {what}" + (f"  [{detail}]" if detail else ""))
    else:
        print(f"  FAIL  {what}" + (f"  [{detail}]" if detail else ""))
        failures += 1


def main() -> int:
    lib = os.environ.get("MODELNEXUS_LIB", "")
    print(f"MODELNEXUS_LIB = {lib}")

    # 1. It loads. On Windows this is where a missing sibling DLL shows up --
    #    the bridge resolves llama/ggml from its own directory, and a staging
    #    bug that dropped one would surface here and nowhere earlier.
    try:
        import modelnexus
    except Exception as e:  # pragma: no cover - the point of the test
        print(f"  FAIL  import modelnexus: {e}")
        return 1

    try:
        version = modelnexus.version()
    except Exception as e:
        print(f"  FAIL  the native library did not load: {e}")
        return 1
    ok(True, "the native library loads", version)
    ok("0.2.0" in version, "it is the 0.2.0 bridge, not a stale asset", version)

    # 2. The 0.2.0 entry points exist. Symbol resolution is lazy in some
    #    bindings, so "it loaded" does not mean "it is complete".
    for name in ("Chat", "Embedder", "set_log_level"):
        ok(hasattr(modelnexus, name), f"binding exposes {name}")

    model = os.environ.get("MODELNEXUS_MODEL", "").strip()
    if not model:
        print()
        print("  SKIP  inference — no model path given.")
        print("        This ran HALF the check: the library loads and reports 0.2.0,")
        print("        but nothing has generated a token on this machine.")
        print("        Re-run as:  verify-windows.cmd C:\\path\\to\\model.gguf")
        print()
        print(f"{checks} checks, {failures} failures (inference SKIPPED)")
        return 1 if failures else 0

    if not os.path.isfile(model):
        print(f"  FAIL  model not found: {model}")
        return 1

    modelnexus.set_log_level(modelnexus.LogLevel.ERROR)

    # 3. Real inference, and a 0.2.0-only call, so a stale native cannot pass.
    with modelnexus.Chat(model, n_ctx=2048) as chat:
        msgs = [{"role": "user", "content": "Name the capital of France in one word."}]

        count = chat.count_tokens(msgs)
        ok(count["tokens"] > 0, "count_tokens works (0.2.0 entry point)", str(count))
        ok(count["n_ctx"] == 2048, "the create config crossed the ABI", f"n_ctx={count['n_ctx']}")

        r = chat.infer(msgs, max_tokens=16, seed=42, temperature=0.0)
        ok("Paris" in r["text"], "inference produces a correct answer", r["text"].strip())

        # Cancellation: the callback returns False on the 3rd token.
        seen = []

        def on_token(piece: str):
            seen.append(piece)
            return False if len(seen) >= 3 else None

        s = chat.infer(
            [{"role": "user", "content": "Count from one to twenty in words."}],
            max_tokens=200,
            seed=7,
            temperature=0.0,
            on_token=on_token,
        )
        ok(s["finish_reason"] == "cancelled", "cancellation works", s["finish_reason"])
        ok(len(seen) == 3, "it stopped at the requested token", f"{len(seen)} tokens")

        # The cache seam, and that the rollback left the engine usable.
        before = chat.cache_status()
        cleared = chat.clear_cache()
        ok(cleared["tokens"] == 0, "clear_cache empties the cache", f"was {before['tokens']}")

        again = chat.infer(msgs, max_tokens=16, seed=42, temperature=0.0)
        ok("Paris" in again["text"], "the engine still works after a clear")

        # Structured output, including the fence strip.
        j = chat.infer(
            [{"role": "user", "content": "Describe Paris."}],
            max_tokens=120,
            seed=42,
            temperature=0.0,
            json_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
                "required": ["city", "country"],
                "additionalProperties": False,
            },
        )
        import json as _json

        try:
            parsed = _json.loads(j["text"])
            ok(set(parsed) == {"city", "country"}, "schema output parses and matches", j["text"].strip())
        except Exception as e:
            ok(False, "schema output parses", f"{e}: {j['text']!r}")

    print()
    print(f"{checks} checks, {failures} failures")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
