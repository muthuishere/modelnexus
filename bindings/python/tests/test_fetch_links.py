"""The downloaded closure must keep its symbolic links.

This is a regression test for a failure that does not look like a failure.

``ZipFile.extractall`` writes a symlink entry as a regular file whose *content* is
the target's name, so ``libllama.0.dylib`` — which the bridge links via ``@rpath`` —
becomes a 23-byte text file. dyld cannot load that, and instead of stopping it
CONTINUES SEARCHING and binds against whatever llama.cpp is installed on the machine.

Observed for real while writing this: a closure pinned to b9371 silently loaded
Homebrew's b9620 and produced correct-looking answers against an engine nobody chose.
On a machine with no system llama.cpp the same closure just fails. Both outcomes are
unacceptable, and neither raises anything.
"""

from __future__ import annotations

import os
import stat
import zipfile
from pathlib import Path

import pytest

from modelnexus import _fetch


def _closure_zip(tmp_path: Path) -> Path:
    """A miniature of the real archive: one real file, one link, one chained link."""
    archive = tmp_path / "natives-test-platform.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("test-platform/libllamabridge.dylib", b"\xcf\xfa\xed\xfe not really mach-o")
        zf.writestr("test-platform/libllama.0.0.9371.dylib", b"real library bytes")

        for name, target in (
            ("test-platform/libllama.0.dylib", "libllama.0.0.9371.dylib"),
            ("test-platform/libllama.dylib", "libllama.0.dylib"),
        ):
            info = zipfile.ZipInfo(name)
            # 0o120000 is S_IFLNK. This is exactly how `zip -y` records a link, and
            # exactly what extractall ignores.
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, target)
    return archive


def test_links_are_recreated_not_written_as_files(tmp_path: Path) -> None:
    archive = _closure_zip(tmp_path)
    dest = tmp_path / "out"
    with zipfile.ZipFile(archive) as zf:
        _fetch._extract_preserving_links(zf, dest)

    d = dest / "test-platform"
    link = d / "libllama.0.dylib"

    assert link.is_symlink(), (
        "the link was written as a regular file; dyld would fall through to whatever "
        "llama.cpp is installed on the machine"
    )
    assert os.readlink(link) == "libllama.0.0.9371.dylib"
    assert link.read_bytes() == b"real library bytes", "the link does not resolve to the real file"

    # Chained links are how llama.cpp actually ships:
    # libfoo.dylib -> libfoo.0.dylib -> libfoo.0.N.dylib
    chained = d / "libllama.dylib"
    assert chained.is_symlink()
    assert chained.read_bytes() == b"real library bytes"


def test_extractall_would_have_broken_it(tmp_path: Path) -> None:
    """Pin the behaviour we are working around, so this test explains itself if
    CPython ever changes it."""
    archive = _closure_zip(tmp_path)
    dest = tmp_path / "naive"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)

    link = dest / "test-platform" / "libllama.0.dylib"
    if link.is_symlink():  # pragma: no cover - would mean CPython changed
        pytest.skip("zipfile now restores symlinks; the workaround may be removable")
    assert link.read_bytes() == b"libllama.0.0.9371.dylib", (
        "extractall wrote the TARGET NAME as the file's content — this is the bug"
    )
