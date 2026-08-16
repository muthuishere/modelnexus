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

from ._lib import (
    EVENT_CB,
    LOG_CB,
    TOKEN_CB,
    NativeLibraryNotFound,
    load,
    platform_key,
    take_string,
)

__all__ = [
    "Chat",
    "LogLevel",
    "set_log_level",
    "set_log_handler",
    "Embedder",
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


class LogLevel:
    """How much the inference engine is allowed to say.

    The bridge defaults to ``WARN`` rather than llama.cpp's own default: a library
    embedded in someone else's process should be quiet unless asked.
    """

    NONE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4


def set_log_level(level: int) -> None:
    """Set how much the engine logs.

    Call before loading a model -- llama.cpp starts logging during load, so
    afterwards is too late to silence it.
    """
    load().llb_set_log_level(int(level))


_log_handler_ref: Any = None


def set_log_handler(handler: Callable[[int, str], None] | None) -> None:
    """Route engine log output to a handler instead of stderr. ``None`` restores stderr."""
    global _log_handler_ref
    lib = load()
    if handler is None:
        lib.llb_set_log_callback(ctypes.cast(None, LOG_CB), None)
        _log_handler_ref = None
        return

    def _on_log(level: int, text: bytes, _user: Any) -> None:
        handler(level, text.decode("utf-8", "replace") if text else "")

    # The core retains this pointer until it is replaced, so it is held at module
    # scope. A local would be collected while native code still holds it.
    _log_handler_ref = LOG_CB(_on_log)
    lib.llb_set_log_callback(_log_handler_ref, None)


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

    ``n_ctx``, ``n_batch`` and ``n_seq_max`` are fixed when the context is built and
    cannot be changed per request, which is why they live here and every generation
    parameter lives on :meth:`infer`. Each is omitted from the request when left
    unset, and the model's default applies -- the values are documented once, on
    ``llb_chat_create`` in ``llamabridge.h``, and are not restated here.
    """

    def __init__(
        self,
        gguf_path: str,
        *,
        n_ctx: int | None = None,
        n_batch: int | None = None,
        n_seq_max: int | None = None,
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

        config: dict[str, Any] = {}
        if n_ctx is not None:
            config["n_ctx"] = n_ctx
        if n_batch is not None:
            config["n_batch"] = n_batch
        if n_seq_max is not None:
            config["n_seq_max"] = n_seq_max
        # No keys means send NULL, not "{}" -- the core's defaults are then reached by
        # the same path they were before this parameter existed.
        config_json = json.dumps(config).encode("utf-8") if config else None

        handle = self._lib.llb_chat_create(
            str(gguf_path).encode("utf-8"), config_json, self._event_cb, None
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
        on_token: Callable[[str], Any] | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        max_tokens: int | None = None,
        repeat_penalty: float | None = None,
        seed: int | None = None,
        stop: Sequence[str] | None = None,
        json_schema: Mapping[str, Any] | None = None,
        grammar: str | None = None,
        reuse_cache: bool | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one turn and return the parsed response.

        Pass ``on_token`` to stream: it is called with each decoded piece as it is
        produced, and the full response is still returned when generation finishes.
        Return ``False`` from it to stop generating -- returning ``None``, which is
        what a callback that only prints does, keeps going. Raising also stops, and
        the exception is re-raised here once the native call has unwound.

        A cancelled turn is a normal response carrying the text produced so far,
        honest usage counts and ``finish_reason == "cancelled"``. It is a result,
        not an error, so it does not raise.

        ``json_schema`` constrains the output to a JSON Schema and the returned text
        is guaranteed to parse; ``grammar`` constrains it to raw GBNF. Supplying both
        is a core-side ``INVALID_REQUEST``, not a precedence rule.

        ``reuse_cache`` defaults to the core's own default (reuse on). Set it False
        when calls must be provably independent -- a determinism harness, or tenants
        sharing one handle. It affects latency, never output.

        Generation parameters (``temperature``, ``top_k``, ``top_p``, ``min_p``,
        ``max_tokens``, ``repeat_penalty``, ``seed``, ``stop``) are named here and
        omitted from the request when left unset, so the core owns every default.

        A misspelled keyword is a ``TypeError`` from Python itself and nothing is
        sent. To reach a core parameter this binding has not named yet, pass
        ``extra={"mirostat": 2}`` -- it goes to the wire unchanged. It is a dict
        rather than ``**kwargs`` for exactly one reason: a dict cannot be reached by
        a typo, so an unnamed parameter stays deliberate.
        """
        self._check_open()

        request: dict[str, Any] = {"messages": list(messages)}
        if tools is not None:
            request["tools"] = list(tools)
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        if temperature is not None:
            request["temperature"] = temperature
        if top_k is not None:
            request["top_k"] = top_k
        if top_p is not None:
            request["top_p"] = top_p
        if min_p is not None:
            request["min_p"] = min_p
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if repeat_penalty is not None:
            request["repeat_penalty"] = repeat_penalty
        if seed is not None:
            request["seed"] = seed
        if stop is not None:
            request["stop"] = list(stop)
        if json_schema is not None:
            request["json_schema"] = dict(json_schema)
        if grammar is not None:
            request["grammar"] = grammar
        if reuse_cache is not None:
            request["reuse_cache"] = reuse_cache
        if extra is not None:
            request.update(extra)
        payload = json.dumps(request).encode("utf-8")

        raised: list[BaseException] = []

        if on_token is None:
            ptr = self._lib.llb_chat_infer(ctypes.c_void_p(self._handle), payload)
        else:
            def _on_token(piece: bytes, _user: Any) -> int:
                try:
                    keep = on_token(piece.decode("utf-8", "replace") if piece else "")
                except BaseException as exc:  # noqa: BLE001 - see below
                    # An exception must not propagate through the C frame: ctypes would
                    # print it and return 0, so the model would keep generating for a
                    # consumer that has already failed. Stop, then re-raise once the
                    # native call has returned.
                    raised.append(exc)
                    return 1
                # Only an explicit False cancels. `not None` is True in Python, so
                # testing truthiness here would cancel every callback that just prints.
                return 1 if keep is False else 0

            cb = TOKEN_CB(_on_token)  # kept alive across the call by this local
            ptr = self._lib.llb_chat_infer_stream(
                ctypes.c_void_p(self._handle), payload, cb, None
            )

        # Take the string first: the core allocated it whatever the callback did, and
        # raising before this point would leak it.
        raw = take_string(self._lib, ptr)
        if raised:
            raise raised[0]

        response = json.loads(raw or "{}")
        if response.get("type") == "error":
            err = response.get("error") or {}
            raise ModelError(err.get("code", "UNKNOWN"), err.get("message", ""))
        return response

    def count_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Count the tokens a message list will occupy, without generating anything.

        Returns ``{"tokens": N, "n_ctx": M}`` -- both, because the number only means
        something against the window it has to fit in.

        Counting needs the model's vocabulary *and* its parsed chat template, neither
        of which a binding holds, which is why this is an ABI call and not arithmetic
        done here. It decodes nothing and does not touch the KV cache, so it is safe
        to call between two inferences.
        """
        self._check_open()

        request: dict[str, Any] = {"messages": list(messages)}
        if tools is not None:
            request["tools"] = list(tools)

        ptr = self._lib.llb_count_tokens(
            ctypes.c_void_p(self._handle), json.dumps(request).encode("utf-8")
        )
        result = json.loads(take_string(self._lib, ptr) or "{}")
        if result.get("type") == "error":
            err = result.get("error") or {}
            raise ModelError(err.get("code", "UNKNOWN"), err.get("message", ""))
        return result

    # -- KV cache ----------------------------------------------------------

    def _cache(self, op: str) -> dict[str, Any]:
        self._check_open()
        ptr = self._lib.llb_chat_cache(
            ctypes.c_void_p(self._handle), json.dumps({"op": op}).encode("utf-8")
        )
        result = json.loads(take_string(self._lib, ptr) or "{}")
        if result.get("type") == "error":
            err = result.get("error") or {}
            raise ModelError(err.get("code", "UNKNOWN"), err.get("message", ""))
        return result

    def cache_status(self) -> dict[str, Any]:
        """What the engine's KV cache currently holds. Changes nothing.

        Returns ``{"tokens": N, "n_ctx": M}`` -- both, because the number only means
        something against the window it has to fit in.
        """
        return self._cache("status")

    def clear_cache(self) -> dict[str, Any]:
        """Drop the KV cache, freeing its memory and forgetting the sequence.

        Returns the state *afterwards*, which is always ``{"tokens": 0, ...}``, so a
        caller can assert the release happened rather than trust that it did.

        Prefix reuse is right for a conversation that appends and wrong when a chat
        moves to unrelated work: the old conversation keeps occupying context memory,
        and two tenants sharing a handle would share a cache. Passing
        ``reuse_cache=False`` on the next inference also clears, but only as a side
        effect of doing work -- no help when the point is to release memory now, or to
        prove the cache is empty before handing the handle on.
        """
        return self._cache("clear")

    # -- LoRA adapters -----------------------------------------------------

    def _lora(self, **op: Any) -> dict[str, Any]:
        self._check_open()
        ptr = self._lib.llb_chat_lora(
            ctypes.c_void_p(self._handle), json.dumps(op).encode("utf-8")
        )
        result = json.loads(take_string(self._lib, ptr) or "{}")
        if result.get("type") == "error":
            err = result.get("error") or {}
            raise ModelError(err.get("code", "UNKNOWN"), err.get("message", ""))
        return result

    def load_lora(self, path: str, scale: float = 1.0) -> int:
        """Load a LoRA adapter and apply it. Returns its id.

        Several adapters can be active at once; they are applied in load order.
        Adapters change *behaviour* -- format, tone, tool-call reliability -- not
        knowledge. For facts, retrieve.
        """
        return int(self._lora(op="load", path=str(path), scale=scale)["id"])

    def set_lora_scale(self, adapter_id: int, scale: float) -> None:
        """Change an adapter's scale. Takes effect on the next inference."""
        self._lora(op="set", id=adapter_id, scale=scale)

    def remove_lora(self, adapter_id: int) -> None:
        """Unload one adapter and reapply the rest."""
        self._lora(op="remove", id=adapter_id)

    def clear_loras(self) -> None:
        """Unload every adapter, returning the model to its base behaviour."""
        self._lora(op="clear")

    def loras(self) -> list[dict[str, Any]]:
        """The adapters currently applied, in order: id, path and scale."""
        return list(self._lora(op="list").get("adapters", []))

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


class Embedder:
    """A model loaded for embedding or reranking.

    Separate from :class:`Chat` because embedding needs a context built with
    embeddings enabled and a pooling strategy fixed at creation -- and reranking
    needs ``pooling="rank"`` specifically, which is what puts the model's
    classification head in the graph. Neither can be switched afterwards.

        with Embedder("bge-small.gguf", pooling="mean") as emb:
            vectors = emb.embed(["hello", "world"])

        with Embedder("bge-reranker.gguf", pooling="rank") as rr:
            for hit in rr.rerank("capital of France?", docs):
                print(hit["index"], hit["score"])

    ``pooling``, ``n_ctx`` and ``n_batch`` are omitted from the request when left
    unset, and the model's default applies. The values are documented once, on
    ``llb_embed_create`` in ``llamabridge.h``, and are not restated here.
    """

    def __init__(
        self,
        gguf_path: str,
        *,
        pooling: str | None = None,
        n_ctx: int | None = None,
        n_batch: int | None = None,
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

        # Held on the instance for the same reason as Chat's: the core keeps the
        # pointer for the life of the handle.
        self._event_cb = EVENT_CB(_on_event)

        config: dict[str, Any] = {}
        if pooling is not None:
            config["pooling"] = pooling
        if n_ctx is not None:
            config["n_ctx"] = n_ctx
        if n_batch is not None:
            config["n_batch"] = n_batch
        # Same rule as Chat: no keys means send NULL, not "{}". A binding that
        # restated the core's default here would pin the old value the day the core
        # moved it -- and Go, sending nothing, would follow while Python did not.
        config_json = json.dumps(config).encode("utf-8") if config else None

        handle = self._lib.llb_embed_create(
            str(gguf_path).encode("utf-8"),
            config_json,
            self._event_cb,
            None,
        )
        if not handle:
            detail = "; ".join(self._events) or "unknown reason"
            raise ModelError("MODEL_LOAD_FAILED", f"could not load {gguf_path} ({detail})")
        self._handle = handle

    def _call(self, fn: Any, request: Mapping[str, Any]) -> dict[str, Any]:
        if not self._handle:
            raise ModelError("ENGINE_CLOSED", "this Embedder has already been closed")
        ptr = fn(ctypes.c_void_p(self._handle), json.dumps(request).encode("utf-8"))
        result = json.loads(take_string(self._lib, ptr) or "{}")
        if result.get("type") == "error":
            err = result.get("error") or {}
            raise ModelError(err.get("code", "UNKNOWN"), err.get("message", ""))
        return result

    def embed(
        self,
        texts: str | Sequence[str],
        *,
        normalize: bool = True,
    ) -> list[list[float]]:
        """Embed one string or many. Returns one vector per input, in input order.

        Normalized to unit length by default, which is what makes a dot product a
        cosine similarity.
        """
        inputs = [texts] if isinstance(texts, str) else list(texts)
        result = self._call(
            self._lib.llb_embed, {"input": inputs, "normalize": normalize}
        )
        return result.get("embeddings", [])

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score documents against a query, best first.

        Each result carries the document's ORIGINAL index, because the list comes
        back reordered. Scores are raw model logits: comparable within one call,
        not across models, and not probabilities.

        Requires a reranker model loaded with ``pooling="rank"``.
        """
        request: dict[str, Any] = {"query": query, "documents": list(documents)}
        if top_n is not None:
            request["top_n"] = top_n
        return self._call(self._lib.llb_rerank, request).get("results", [])

    def close(self) -> None:
        """Release the model and its context. Idempotent."""
        if self._handle:
            self._lib.llb_embed_destroy(ctypes.c_void_p(self._handle))
            self._handle = None

    def __enter__(self) -> "Embedder":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        try:
            self.close()
        except Exception:
            pass
