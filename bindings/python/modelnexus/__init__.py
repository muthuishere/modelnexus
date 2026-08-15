"""modelnexus -- local LLM inference, in your own process.

    import modelnexus

    with modelnexus.Chat("qwen2.5-1.5b-instruct-q4_k_m.gguf") as chat:
        reply = chat.infer(messages=[{"role": "user", "content": "hello"}])
        print(reply["text"])

No server, no subprocess, no port. The model runs inside this interpreter.
"""

from __future__ import annotations

import ctypes
import json
from typing import Any, Callable, Iterable, Mapping, Sequence

from ._lib import EVENT_CB, TOKEN_CB, NativeLibraryNotFound, load, platform_key, take_string

__all__ = [
    "Chat",
    "ModelError",
    "ToolsUnsupportedError",
    "NativeLibraryNotFound",
    "model_info",
    "version",
    "platform_key",
]


class ModelError(RuntimeError):
    """An inference or model-loading failure reported by the core.

    ``code`` is the core's stable machine-readable identifier and is identical
    across every language binding; the message is for humans.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}" if code else message)
        self.code = code
        self.message = message


class ToolsUnsupportedError(ModelError):
    """The model's chat template cannot do tool calling, so it was rejected at load.

    This is a deliberate contract, not a limitation: a model that silently degrades
    tool calls to prose is worse than one that refuses to load.
    """

    def __init__(self, path: str) -> None:
        super().__init__(
            "MODEL_NOT_TOOL_CAPABLE",
            f"{path} has no tool-calling chat template",
        )


def version() -> str:
    """Bridge version and the llama.cpp tag it was linked against."""
    lib = load()
    raw = lib.llb_version()
    return raw.decode("utf-8") if raw else ""


def model_info(gguf_path: str) -> dict[str, Any]:
    """Inspect a GGUF's tool-calling capability without loading an engine.

    Cheap enough to call before committing to a multi-gigabyte load, which is the
    entire reason the core exposes it separately.
    """
    lib = load()
    ptr = lib.llb_model_info(str(gguf_path).encode("utf-8"))
    return json.loads(take_string(lib, ptr) or "{}")


class Chat:
    """A loaded model plus its inference context.

    Holds a native handle, so it must be closed. Use it as a context manager, or
    call :meth:`close` yourself.
    """

    def __init__(
        self,
        gguf_path: str,
        *,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self._lib = load()
        self._handle: int | None = None
        self._events: list[str] = []

        def _on_event(event: bytes, _user: Any) -> None:
            text = event.decode("utf-8", "replace") if event else ""
            self._events.append(text)
            if on_event is not None:
                on_event(text)

        # The core STORES this pointer on the engine (llamabridge.cpp:540) and calls it
        # for the whole life of the handle -- not just during create. So the trampoline
        # must outlive this constructor: hold it on the instance, not in a local. A
        # local here is collected as soon as __init__ returns and the next event emitted
        # by the core jumps into freed memory. Verified: it segfaults.
        self._event_cb = EVENT_CB(_on_event)
        handle = self._lib.llb_chat_create(
            str(gguf_path).encode("utf-8"), self._event_cb, None
        )

        if not handle:
            # NULL is the one place the core does signal failure with a null pointer;
            # the reason arrives through the event callback instead.
            if any("tools_unsupported" in e for e in self._events):
                raise ToolsUnsupportedError(str(gguf_path))
            detail = "; ".join(self._events) or "unknown reason"
            raise ModelError("MODEL_LOAD_FAILED", f"could not load {gguf_path} ({detail})")

        self._handle = handle

    # -- inference ---------------------------------------------------------

    def infer(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Iterable[Mapping[str, Any]] | None = None,
        tool_choice: str | None = None,
        on_token: Callable[[str], None] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Run one turn and return the parsed response.

        Pass ``on_token`` to stream: it is called with each decoded piece as it is
        produced, and the full response is still returned when generation finishes.

        Generation parameters (``temperature``, ``top_k``, ``top_p``, ``min_p``,
        ``max_tokens``, ``repeat_penalty``, ``seed``, ``stop``) travel inside the
        request JSON, so new ones need no change here.
        """
        self._check_open()

        request: dict[str, Any] = {"messages": list(messages)}
        if tools is not None:
            request["tools"] = list(tools)
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        request.update(params)
        payload = json.dumps(request).encode("utf-8")

        if on_token is None:
            ptr = self._lib.llb_chat_infer(ctypes.c_void_p(self._handle), payload)
        else:
            def _on_token(piece: bytes, _user: Any) -> None:
                if piece:
                    on_token(piece.decode("utf-8", "replace"))

            cb = TOKEN_CB(_on_token)  # kept alive across the call by this local
            ptr = self._lib.llb_chat_infer_stream(
                ctypes.c_void_p(self._handle), payload, cb, None
            )

        response = json.loads(take_string(self._lib, ptr) or "{}")
        if response.get("type") == "error":
            err = response.get("error") or {}
            raise ModelError(err.get("code", "UNKNOWN"), err.get("message", ""))
        return response

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Release the model and its context. Idempotent."""
        if self._handle:
            self._lib.llb_chat_destroy(ctypes.c_void_p(self._handle))
            self._handle = None

    def _check_open(self) -> None:
        if not self._handle:
            raise ModelError("ENGINE_CLOSED", "this Chat has already been closed")

    def __enter__(self) -> "Chat":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        try:
            self.close()
        except Exception:
            pass
