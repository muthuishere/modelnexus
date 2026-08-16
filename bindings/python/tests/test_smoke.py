"""Python binding tests.

Mirrors bindings/go/modelnexus_test.go case for case. The two suites asserting the
same behaviour -- and the same error codes -- is what keeps the bindings from
drifting; that is the whole point of a shared C core.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modelnexus as mx  # noqa: E402


def model() -> str:
    """The GGUF to test against, or skip.

    Skipping rather than failing is deliberate: the binding is worth testing on a
    machine with no multi-gigabyte model sitting around.
    """
    path = os.environ.get("MODELNEXUS_MODEL", "")
    if not path:
        pytest.skip("set MODELNEXUS_MODEL to a tool-capable GGUF to run this test")
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        pytest.skip(f"MODELNEXUS_MODEL={path} is not readable")
    return path


def test_version_identifies_bridge_and_engine():
    v = mx.version()
    assert "llamabridge" in v
    assert "llama.cpp" in v


def test_info_reports_tool_capability():
    info = mx.model_info(model())
    assert info["supports_tools"] is True
    assert info["chat_format"]


def test_infer():
    with mx.Chat(model()) as chat:
        r = chat.infer(
            [{"role": "user", "content": "Reply with exactly: pong"}],
            max_tokens=16,
            seed=1,
        )
    assert r["type"] == "assistant_text"
    assert r["text"]
    assert r["usage"]["total_tokens"] > 0


def test_infer_stream_delivers_tokens_and_final_response():
    pieces: list[str] = []
    with mx.Chat(model()) as chat:
        r = chat.infer(
            [{"role": "user", "content": "Count: 1 2 3"}],
            max_tokens=24,
            seed=1,
            on_token=pieces.append,
        )
    assert pieces, "expected streamed token pieces"
    # Streaming must not be a separate code path with a different result.
    assert r["usage"]["completion_tokens"] > 0


def test_event_handler_survives_the_call():
    # The core stores the event callback on the engine and keeps calling it, so a
    # binding that lets the trampoline die after __init__ segfaults later. This test
    # exists because that is exactly what happened during development -- see
    # spikes/0002-callback-lifetime-and-threading.
    events: list[str] = []
    with mx.Chat(model(), on_event=events.append) as chat:
        chat.infer([{"role": "user", "content": "hi"}], max_tokens=8)
    assert events


def test_missing_model_is_a_typed_error():
    with pytest.raises(mx.ModelError) as excinfo:
        mx.Chat("/definitely/not/here.gguf")
    # The code is part of the cross-binding contract: Go reports the same one.
    assert excinfo.value.code == "MODEL_LOAD_FAILED"


def test_use_after_close_is_rejected():
    chat = mx.Chat(model())
    chat.close()
    chat.close()  # idempotent
    with pytest.raises(mx.ModelError) as excinfo:
        chat.infer([{"role": "user", "content": "hi"}])
    assert excinfo.value.code == "ENGINE_CLOSED"


# ---------------------------------------------------------------------------
# Inference control (ADR-0008). Mirrors core/tests/abi_test.c case for case.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chat():
    """One engine shared by the inference-control tests.

    Shared deliberately, not just to save a load: cache reuse and cancellation
    rollback are properties of a handle ACROSS calls, and a fresh engine per test
    would assert nothing about either.
    """
    with mx.Chat(model(), n_ctx=4096, n_batch=512) as c:
        yield c


CAPITAL = [{"role": "user", "content": "Name the capital of France in one word."}]


def test_no_config_still_loads():
    # The config parameter is new in 0.2.0; omitting it must reach exactly the
    # defaults the old two-argument signature used.
    with mx.Chat(model()) as c:
        assert c.count_tokens(CAPITAL)["n_ctx"] > 0


def test_count_tokens_reports_a_count_and_the_window(chat):
    counted = chat.count_tokens(CAPITAL)
    assert counted["type"] == "token_count"
    assert counted["tokens"] > 0
    # A short user turn plus a chat template is small; if this is thousands, the
    # template was not applied or the wrong thing was tokenized.
    assert counted["tokens"] < 200
    assert counted["n_ctx"] >= counted["tokens"]


def test_reuse_cache_does_not_change_output(chat):
    # Reuse is a latency property. Any observable difference in output is a defect,
    # and this is the assertion that catches it.
    params = dict(max_tokens=16, seed=42, temperature=0.0)
    cold = chat.infer(CAPITAL, reuse_cache=False, **params)
    warm = chat.infer(CAPITAL, reuse_cache=True, **params)
    cold_again = chat.infer(CAPITAL, reuse_cache=False, **params)

    assert warm["text"] == cold["text"]
    assert cold_again["text"] == cold["text"], "repeated cold runs are the control"


def test_returning_false_cancels_and_is_not_an_error(chat):
    pieces: list[str] = []

    def stop_after_eight(piece: str):
        pieces.append(piece)
        return False if len(pieces) >= 8 else None

    r = chat.infer(
        [{"role": "user", "content": "Count slowly from one to fifty in words, one per line."}],
        max_tokens=300,
        seed=7,
        temperature=0.0,
        on_token=stop_after_eight,
    )

    # A cancelled generation is a result, not an exception: it produced real text and
    # burned real tokens, and the caller is owed both.
    assert r["finish_reason"] == "cancelled"
    assert len(pieces) == 8, "generation stopped at the requested token, not later"
    assert r["usage"]["completion_tokens"] == 8


def test_a_request_after_a_cancellation_is_still_correct(chat):
    # The D2xD4 interaction: an abort leaves a partial assistant turn in the cache,
    # and without rollback the next call's prefix match extends a truncated turn as
    # though it were complete. Silent, plausible, wrong -- so assert the ANSWER.
    seen = 0

    def stop_after_five(_piece: str):
        nonlocal seen
        seen += 1
        return False if seen >= 5 else None

    chat.infer(
        [{"role": "user", "content": "Count slowly from one to fifty in words, one per line."}],
        max_tokens=300,
        seed=7,
        temperature=0.0,
        on_token=stop_after_five,
    )

    after = chat.infer(CAPITAL, max_tokens=16, seed=42, temperature=0.0)
    assert "Paris" in after["text"]


def test_a_callback_returning_none_does_not_cancel(chat):
    # The trap this exists for: `not None` is True, so a binding that tests the
    # callback's result for truthiness cancels every stream that just prints.
    pieces: list[str] = []
    r = chat.infer(CAPITAL, max_tokens=16, seed=42, temperature=0.0, on_token=pieces.append)
    assert pieces
    assert r["finish_reason"] != "cancelled"


def test_a_raising_callback_stops_and_re_raises(chat):
    seen = 0

    def blow_up(_piece: str) -> None:
        nonlocal seen
        seen += 1
        raise ValueError("consumer went away")

    with pytest.raises(ValueError, match="consumer went away"):
        chat.infer(
            [{"role": "user", "content": "Count slowly from one to fifty in words, one per line."}],
            max_tokens=300,
            seed=7,
            temperature=0.0,
            on_token=blow_up,
        )
    # Raised on the first piece and generation stopped there, rather than the C frame
    # swallowing the exception and running to max_tokens.
    assert seen == 1


def test_schema_constrained_output_parses_and_matches_the_schema(chat):
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"],
        "additionalProperties": False,
    }
    r = chat.infer(
        [{"role": "user", "content": "Describe Paris."}],
        max_tokens=120,
        seed=42,
        temperature=0.0,
        json_schema=schema,
    )

    # json.loads, never a substring check: upstream's grammar PERMITS a ```json fence,
    # and a fence is exactly the failure an `in` assertion cannot see.
    value = json.loads(r["text"])
    assert set(value) == {"city", "country"}
    assert isinstance(value["city"], str)
    assert isinstance(value["country"], str)


def test_a_raw_gbnf_grammar_constrains_generation(chat):
    r = chat.infer(
        [{"role": "user", "content": "Pick a colour."}],
        max_tokens=16,
        seed=42,
        temperature=0.0,
        grammar='root ::= "red" | "blue"',
    )
    assert r["text"].strip() in ("red", "blue")


def test_schema_and_grammar_together_is_rejected(chat):
    with pytest.raises(mx.ModelError) as excinfo:
        chat.infer(
            [{"role": "user", "content": "hi"}],
            json_schema={"type": "object"},
            grammar='root ::= "x"',
        )
    # An error rather than a precedence rule: a silent winner between two output
    # constraints is a debugging session nobody should have.
    assert excinfo.value.code == "INVALID_REQUEST"


def lora_pair() -> tuple[str, str]:
    """A base model and a LoRA adapter built for it, or skip.

    A LoRA is architecture-specific: it only loads against the base it was trained
    on, so this needs its own matched pair rather than reusing MODELNEXUS_MODEL.
    """
    base = os.path.expanduser(os.environ.get("MODELNEXUS_LORA_BASE", ""))
    lora = os.path.expanduser(os.environ.get("MODELNEXUS_LORA", ""))
    if not base or not lora:
        pytest.skip("set MODELNEXUS_LORA_BASE and MODELNEXUS_LORA to run this test")
    if not os.path.isfile(base) or not os.path.isfile(lora):
        pytest.skip("MODELNEXUS_LORA_BASE / MODELNEXUS_LORA are not both readable")
    return base, lora


def test_lora_full_lifecycle_with_a_real_adapter():
    base, lora = lora_pair()
    with mx.Chat(base) as chat:
        assert chat.loras() == []

        first = chat.load_lora(lora, scale=1.0)
        applied = chat.loras()
        assert len(applied) == 1
        assert applied[0]["id"] == first
        assert applied[0]["scale"] == 1.0

        chat.set_lora_scale(first, 0.5)
        assert chat.loras()[0]["scale"] == 0.5

        # Several adapters can be active at once; ids are stable and independent.
        second = chat.load_lora(lora, scale=0.25)
        assert [a["id"] for a in chat.loras()] == [first, second]

        chat.remove_lora(second)
        assert [a["id"] for a in chat.loras()] == [first]

        # Generation must still work with an adapter applied -- the point of loading it.
        r = chat.infer([{"role": "user", "content": "Say the word: apple"}], max_tokens=12, seed=1)
        assert r["type"] == "assistant_text"
        assert r["text"]

        chat.clear_loras()
        assert chat.loras() == []

        r2 = chat.infer([{"role": "user", "content": "Say the word: apple"}], max_tokens=12, seed=1)
        assert r2["type"] == "assistant_text"
