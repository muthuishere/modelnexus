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
