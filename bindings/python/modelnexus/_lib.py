"""Locating and loading the native bridge.

The core is a shared library that has to be found at runtime. Every binding needs
the same three-step search and the same explicit failure when it comes up empty --
"library not found" must never surface as a segfault or a bare OSError from ctypes.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path

_LIB_BASENAME = "llamabridge"


class NativeLibraryNotFound(RuntimeError):
    """The native bridge could not be located or loaded.

    Raised at first use rather than at import, so that merely importing modelnexus
    in an environment without the native library is not itself an error.
    """


def platform_key() -> str:
    system = platform.system().lower()
    if system == "darwin":
        os_key = "darwin"
    elif system == "linux":
        os_key = "linux"
    elif system == "windows":
        os_key = "windows"
    else:  # pragma: no cover - exercised only on exotic platforms
        raise NativeLibraryNotFound(f"unsupported OS: {platform.system()}")

    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch_key = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch_key = "aarch64"
    else:  # pragma: no cover
        raise NativeLibraryNotFound(f"unsupported arch: {platform.machine()}")

    return f"{os_key}-{arch_key}"


def _lib_filename() -> str:
    if sys.platform == "darwin":
        return f"lib{_LIB_BASENAME}.dylib"
    if sys.platform == "win32":
        return f"{_LIB_BASENAME}.dll"
    return f"lib{_LIB_BASENAME}.so"


def _candidate_paths() -> list[Path]:
    """Search order, most explicit first.

    1. ``MODELNEXUS_LIB`` -- a full path to the library, or the directory holding it.
    2. Natives bundled inside this package (how a wheel ships).
    3. The repo's own ``core/dist/<platform>`` (how a developer works).
    """
    name = _lib_filename()
    key = platform_key()
    out: list[Path] = []

    env = os.environ.get("MODELNEXUS_LIB")
    if env:
        p = Path(env).expanduser()
        out.append(p if p.is_file() else p / name)

    here = Path(__file__).resolve().parent
    out.append(here / "native" / key / name)

    # bindings/python/modelnexus/_lib.py -> repo root
    repo = here.parent.parent.parent
    out.append(repo / "core" / "dist" / key / name)

    return out


_cached: ctypes.CDLL | None = None


def load() -> ctypes.CDLL:
    """Load the bridge, or raise NativeLibraryNotFound naming everywhere we looked."""
    global _cached
    if _cached is not None:
        return _cached

    tried: list[str] = []
    for path in _candidate_paths():
        tried.append(str(path))
        if not path.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            # Present but unloadable is a different failure from absent, and much
            # more confusing to debug -- say which one happened.
            raise NativeLibraryNotFound(
                f"found the native bridge at {path} but could not load it: {exc}"
            ) from exc
        _bind_signatures(lib)
        _cached = lib
        return lib

    raise NativeLibraryNotFound(
        "could not locate the modelnexus native bridge.\n"
        "Looked in:\n  " + "\n  ".join(tried) + "\n"
        "Build it with core/build.sh, or set MODELNEXUS_LIB to the library or its directory."
    )


def _bind_signatures(lib: ctypes.CDLL) -> None:
    """Declare argtypes/restypes.

    restype is deliberately c_void_p, never c_char_p, for every function that returns
    a string the caller owns: ctypes converts c_char_p to bytes and discards the
    pointer, which would make llb_string_free impossible to call and leak every result.
    """
    lib.llb_version.argtypes = []
    lib.llb_version.restype = ctypes.c_char_p  # static, owned by the bridge -- never freed

    lib.llb_model_info.argtypes = [ctypes.c_char_p]
    lib.llb_model_info.restype = ctypes.c_void_p

    lib.llb_chat_create.argtypes = [ctypes.c_char_p, EVENT_CB, ctypes.c_void_p]
    lib.llb_chat_create.restype = ctypes.c_void_p

    lib.llb_chat_infer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.llb_chat_infer.restype = ctypes.c_void_p

    lib.llb_chat_infer_stream.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        TOKEN_CB,
        ctypes.c_void_p,
    ]
    lib.llb_chat_infer_stream.restype = ctypes.c_void_p

    lib.llb_chat_lora.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.llb_chat_lora.restype = ctypes.c_void_p

    lib.llb_embed_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, EVENT_CB, ctypes.c_void_p]
    lib.llb_embed_create.restype = ctypes.c_void_p

    lib.llb_embed.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.llb_embed.restype = ctypes.c_void_p

    lib.llb_rerank.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.llb_rerank.restype = ctypes.c_void_p

    lib.llb_embed_destroy.argtypes = [ctypes.c_void_p]
    lib.llb_embed_destroy.restype = None

    lib.llb_string_free.argtypes = [ctypes.c_void_p]
    lib.llb_string_free.restype = None

    lib.llb_chat_destroy.argtypes = [ctypes.c_void_p]
    lib.llb_chat_destroy.restype = None


# void (*)(const char* event, void* user_data)
EVENT_CB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)
# void (*)(const char* token_piece, void* user_data)
TOKEN_CB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


def take_string(lib: ctypes.CDLL, ptr: int | None) -> str:
    """Copy a core-owned string into Python and free the original.

    Every string the core returns is heap-allocated and ours to release. Doing the
    copy and the free in one place is what keeps that from being a leak waiting to
    be forgotten at each call site.
    """
    if not ptr:
        return ""
    try:
        return ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")  # type: ignore[union-attr]
    finally:
        lib.llb_string_free(ctypes.c_void_p(ptr))
