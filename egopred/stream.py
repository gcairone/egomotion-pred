"""Streaming reads of remote .tgz archives: members are processed as they arrive, never
saved whole to disk. Lets us keep 1 of 8 cameras while paying only for that on disk."""

import logging
import tarfile
import time
import urllib.request
from collections.abc import Iterator
from typing import BinaryIO

log = logging.getLogger(__name__)

USER_AGENT = "egopred/0.1"


class ProgressReader:
    """File-like wrapper that counts bytes read and logs throughput periodically."""

    def __init__(self, raw: BinaryIO, label: str, total: int | None = None, every: float = 30.0):
        self._raw = raw
        self.label = label
        self.total = total
        self.nbytes = 0
        self._t0 = self._last = time.monotonic()
        self._every = every

    def read(self, n: int = -1) -> bytes:
        chunk = self._raw.read(n)
        self.nbytes += len(chunk)
        now = time.monotonic()
        if now - self._last >= self._every:
            self._last = now
            rate = self.nbytes / (now - self._t0) / 1e6
            pct = f" ({100 * self.nbytes / self.total:.0f}%)" if self.total else ""
            log.info("%s: %.2f GB read%s at %.1f MB/s", self.label, self.nbytes / 1e9, pct, rate)
        return chunk

    def close(self) -> None:
        self._raw.close()


def open_source(src: str, timeout: float = 120.0) -> ProgressReader:
    """Open a URL or a local path for sequential reading."""
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": USER_AGENT})
        resp = urllib.request.urlopen(req, timeout=timeout)
        total = resp.headers.get("Content-Length")
        return ProgressReader(resp, src.rsplit("/", 1)[-1], int(total) if total else None)
    return ProgressReader(open(src, "rb"), src.rsplit("/", 1)[-1])


def iter_tgz(fileobj) -> Iterator[tuple[str, int, BinaryIO]]:
    """Yield (name, size, reader) for each regular file of a gzip'd tar stream.

    The reader is only valid until the next item is requested (stream mode)."""
    with tarfile.open(fileobj=fileobj, mode="r|gz") as tf:
        for member in tf:
            if member.isfile():
                yield member.name, member.size, tf.extractfile(member)


def download(url: str, dest, timeout: float = 120.0) -> int:
    """Download `url` to `dest` atomically (temp file + rename). Returns the size in bytes."""
    from pathlib import Path

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest.stat().st_size
