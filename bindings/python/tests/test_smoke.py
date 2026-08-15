"""Python binding tests.

Mirrors bindings/go/modelnexus_test.go case for case. The two suites asserting the
same behaviour -- and the same error codes -- is what keeps the bindings from
drifting; that is the whole point of a shared C core.
"""

from __future__ import annotations

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
