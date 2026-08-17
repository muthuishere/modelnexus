"""Download the native closure when this wheel does not carry one.

The wheel is the happy path and stays the happy path: on a platform we publish, the
library is already inside the package and this module is never reached. It exists for
the platform we do *not* publish — the user who would otherwise get an import error
with no way forward.

Kept deliberately close to the Go binding's ``fetch.go``: same release, same URL, same
cache layout, so two bindings on one machine share a download instead of each keeping
its own copy. That is a requirement, not a nicety — see ADR-0010 and the model-core
spec's ordered resolution steps.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# The llama.cpp release this binding's native library is built against, and the C ABI
# it speaks. BOTH belong in the cache key: keyed on the llama.cpp tag alone, a cache
# populated by an older binding is silently reused by a newer one that needs entry
# points it does not have. The Go binding had exactly that bug.
LLAMA_TAG = "b9371"
BRIDGE_VERSION = "0.2.1"

_NATIVES_REPO = "muthuishere/modelnexus"


class NativesUnavailable(RuntimeError):
    """The natives could not be downloaded, with a reason a human can act on."""


def cache_dir() -> Path:
    """Where downloaded natives live. ``MODELNEXUS_CACHE`` overrides the root."""
    env = os.environ.get("MODELNEXUS_CACHE")
    base = Path(env).expanduser() if env else _user_cache_dir() / "modelnexus"
    return base / f"{LLAMA_TAG}-{BRIDGE_VERSION}"


def _user_cache_dir() -> Path:
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local)
    elif os.sys.platform == "darwin":  # type: ignore[attr-defined]
        return Path.home() / "Library" / "Caches"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg) if xdg else Path.home() / ".cache"


def fetch(platform_key: str, lib_name: str) -> Path:
    """Download and unpack the closure for ``platform_key``; return its directory.

    Idempotent: an already-populated cache is returned without a network call.
    """
    root = cache_dir()
    target = root / platform_key
    if (target / lib_name).is_file():
        return target

    url = (
        f"https://github.com/{_NATIVES_REPO}/releases/download/"
        f"natives-{LLAMA_TAG}/natives-{platform_key}.zip"
    )

    try:
        with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed https URL
            if resp.status != 200:
                raise NativesUnavailable(_no_build_message(url, resp.status, platform_key))
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        raise NativesUnavailable(_no_build_message(url, exc.code, platform_key)) from exc
    except urllib.error.URLError as exc:
        raise NativesUnavailable(
            f"could not reach {url}: {exc.reason}.\n"
            "If this machine has no network, build the library with core/build.sh and "
            "set MODELNEXUS_LIB to the directory holding it."
        ) from exc

    root.mkdir(parents=True, exist_ok=True)
    # Unpack beside the target and RENAME. A half-unpacked directory at the final path
    # would look like a valid cache on the next run and fail deep inside the loader.
    staging = Path(tempfile.mkdtemp(dir=root, prefix=".part-"))
    try:
        with tempfile.NamedTemporaryFile(dir=root, suffix=".zip", delete=False) as tmp:
            tmp.write(payload)
            archive = Path(tmp.name)
        try:
            with zipfile.ZipFile(archive) as zf:
                _extract_preserving_links(zf, staging)
        finally:
            archive.unlink(missing_ok=True)

        # The archive contains a single <platform-key>/ directory.
        inner = staging / platform_key
        unpacked = inner if inner.is_dir() else staging
        if not (unpacked / lib_name).is_file():
            raise NativesUnavailable(
                f"{url} unpacked without {lib_name} in it — the published archive is broken"
            )

        # zipfile does not restore the executable bit, and some loaders care.
        for entry in unpacked.iterdir():
            # is_symlink() first: chmod follows links, and a link to a name that does
            # not exist yet would raise.
            if not entry.is_symlink() and entry.is_file():
                entry.chmod(0o755)

        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        unpacked.replace(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return target


def _extract_preserving_links(zf: zipfile.ZipFile, dest: Path) -> None:
    """Unpack, recreating symbolic links instead of writing files full of paths.

    ``ZipFile.extractall`` does not restore symlinks: it writes each one as a regular
    file whose *content* is the target's name. The closure has 18 of them, so that
    turns ``libllama.0.dylib`` -- which the bridge links via ``@rpath`` -- into a
    23-byte text file.

    The failure is worse than a crash. dyld cannot load the stub, and instead of
    stopping it CONTINUES ITS SEARCH and silently binds against whatever llama.cpp is
    installed on the machine. Observed for real: a closure pinned to b9371 loaded
    Homebrew's b9620 and produced correct-looking output against an engine nobody
    chose. On a machine with no llama.cpp installed the same closure simply fails.

    Go's Fetch already got this right (``fetch.go:156``); this is the same rule.
    """
    links: list[tuple[Path, str]] = []
    for info in zf.infolist():
        out = dest / info.filename
        if stat.S_ISLNK(info.external_attr >> 16):
            # Deferred: a link may point at a name not yet written, and llama.cpp
            # chains them (libggml.dylib -> libggml.0.dylib -> libggml.0.13.0.dylib).
            links.append((out, zf.read(info).decode()))
            continue
        if info.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)

    for path, target in links:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            path.unlink()
        path.symlink_to(target)


def _no_build_message(url: str, status: int, platform_key: str) -> str:
    return (
        f"no published native library for {platform_key} ({url} returned HTTP {status}).\n"
        f"Build it yourself with core/build.sh and set MODELNEXUS_LIB to core/dist/{platform_key}."
    )
