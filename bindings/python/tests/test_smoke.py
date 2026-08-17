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


def test_a_misspelled_parameter_is_rejected_before_anything_is_sent(chat):
    # `max_token` -- singular -- used to be marshalled, sent, ignored by the core, and
    # the call succeeded on the default 256. Go would not compile it; Python must not
    # accept it. The name has to appear in the message or the caller is still hunting.
    before = chat.cache_status()["tokens"]
    with pytest.raises(TypeError, match="max_token"):
        chat.infer(CAPITAL, max_token=64)
    # Nothing reached the core: the KV cache is exactly where it was.
    assert chat.cache_status()["tokens"] == before


def test_extra_reaches_the_wire_unchanged(chat):
    # The escape hatch ADR-0002 requires: a core parameter this binding has not named
    # costs no binding change. A dict, not **kwargs, because a dict cannot be typed by
    # accident. If `extra` were dropped the core would never see the key -- an unknown
    # one is ignored, so assert with a key the core acts on instead.
    r = chat.infer(CAPITAL, seed=42, temperature=0.0, extra={"max_tokens": 4})
    assert r["usage"]["completion_tokens"] <= 4

    # And an unknown key still crosses, rather than being filtered against a list the
    # binding would have to keep in step with the core.
    r2 = chat.infer(CAPITAL, max_tokens=8, seed=42, temperature=0.0, extra={"mirostat": 2})
    assert r2["type"] == "assistant_text"


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


def test_cache_status_and_clear(chat):
    # The assertion that matters is that the clear is OBSERVABLE. A clear that
    # silently did nothing would still return a well-formed response, and the next
    # inference would still be correct -- just slow, and still holding the previous
    # tenant's conversation. So: infer, see a non-zero cache, clear, see zero, then
    # infer again and prove the handle still works.
    chat.infer(CAPITAL, max_tokens=16, seed=42, temperature=0.0)

    before = chat.cache_status()
    assert before["type"] == "cache"
    assert before["tokens"] > 0
    assert before["n_ctx"] >= before["tokens"]

    # Status is the non-destructive call -- the binding's stand-in for the ABI's
    # "a NULL request reads status, it does not clear". Reading twice must not empty
    # the cache; getting that backwards would make an innocent-looking call wipe a
    # conversation.
    assert chat.cache_status()["tokens"] == before["tokens"]

    assert chat.clear_cache()["tokens"] == 0
    assert chat.cache_status()["tokens"] == 0, "the clear did not persist"

    after = chat.infer(CAPITAL, max_tokens=16, seed=42, temperature=0.0)
    assert "Paris" in after["text"]


def test_cache_on_a_closed_chat_is_rejected():
    c = mx.Chat(model())
    c.close()
    for call in (c.cache_status, c.clear_cache):
        with pytest.raises(mx.ModelError) as excinfo:
            call()
        assert excinfo.value.code == "ENGINE_CLOSED"


# ---------------------------------------------------------------------------
# Create config. Asserted through the event the CORE emits, which is what the
# parity gate reads -- self-reporting what a binding "meant to send" would pass
# while the wire disagreed.
# ---------------------------------------------------------------------------


def created_config(cls, **kw) -> str:
    seen: list[str] = []

    def on_event(e: str) -> None:
        if e.startswith("create_config:"):
            seen.append(e[len("create_config:"):])

    engine = cls(model(), on_event=on_event, **kw)
    engine.close()
    assert seen, "the core did not report a create config"
    return seen[-1]


def test_an_embedder_with_no_options_sends_no_config():
    # It used to send {"n_batch":512} while Go sent nothing -- same core, same intent,
    # different request. Harmless only until the core moves the default, at which
    # point Python would pin the old value and Go would follow the new one.
    assert json.loads(created_config(mx.Embedder)) is None


def test_an_embedder_sends_only_what_was_asked_for():
    assert json.loads(created_config(mx.Embedder, pooling="mean")) == {"pooling": "mean"}


def test_a_chat_with_no_options_sends_no_config():
    assert json.loads(created_config(mx.Chat)) is None


def test_a_chat_sends_only_what_was_asked_for():
    assert json.loads(created_config(mx.Chat, n_ctx=2048)) == {"n_ctx": 2048}


def test_gpu_layers_is_absent_unless_asked_for():
    # Unset must not reach the core at all: the core's default is every layer, and a
    # binding that sent a number here would pin it the day the core moves.
    assert json.loads(created_config(mx.Chat)) is None


def test_gpu_layers_zero_is_sent_rather_than_read_as_unset():
    # The assertion that matters. 0 means "CPU only" and is a deliberate request; a
    # truthiness check would drop it and quietly hand the caller the GPU instead.
    assert json.loads(created_config(mx.Chat, n_gpu_layers=0)) == {"n_gpu_layers": 0}
    assert json.loads(created_config(mx.Embedder, n_gpu_layers=0)) == {"n_gpu_layers": 0}


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
