"""Front-camera ingest: stream an all-camera OpenScene/NAVSIM archive and keep only CAM_F0.

Output per source archive: an uncompressed tar shard of JPEGs named `<log>/CAM_F0/<id>.jpg`
(the same relative path the metadata stores in `f0_path`) plus an index parquet with the
byte offset of every member, so loaders can read a frame without scanning the tar.
"""

import http.client
import io
import logging
import re
import tarfile
import time
import zlib
from pathlib import Path

import pandas as pd
from PIL import Image

from .. import paths
from ..manifest import Manifest
from ..stream import iter_tgz, open_source

log = logging.getLogger(__name__)

# A long HTTP stream can be cut mid-archive ("unexpected end of data"): such errors are
# retried from scratch, anything else (e.g. a corrupt frame) fails immediately.
STREAM_ERRORS = (tarfile.ReadError, EOFError, zlib.error, OSError, http.client.HTTPException)
STREAM_ATTEMPTS = 3
RETRY_WAIT_S = 30.0

# Quality of re-encoded (downscaled) frames: at 960x540 q85 averages ~87 KB per frame, so
# the ~170k front frames of navtrain + navtest take ~15 GB.
JPEG_QUALITY = 85

FRONT_RE = re.compile(r"(?:^|/)([\w.\-]+)/CAM_F0/([0-9a-fA-F]+)\.jpg$")


def front_relpath(member_name: str) -> str | None:
    m = FRONT_RE.search(member_name)
    return None if m is None else f"{m.group(1)}/CAM_F0/{m.group(2)}.jpg"


def resize_jpeg(data: bytes, max_width: int, quality: int = JPEG_QUALITY) -> bytes:
    im = Image.open(io.BytesIO(data))
    if im.width <= max_width:
        return data
    size = (max_width, round(im.height * max_width / im.width))
    im.draft("RGB", size)  # DCT-domain downscale: much faster decode
    im = im.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def shard_paths(split: str, archive: str) -> tuple[Path, Path]:
    base = paths.OPENSCENE_FRAMES / split / archive
    return base.with_suffix(".tar"), base.with_suffix(".index.parquet")


def extract_front(fileobj, out_tar: Path, wanted: set[str] | None = None,
                  max_width: int | None = None) -> dict:
    """Copy CAM_F0 JPEGs from a .tgz stream into `out_tar` (written atomically)."""
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_tar.with_name(out_tar.name + ".partial")
    n = nbytes = 0
    logs: set[str] = set()
    with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as out:
        for name, _size, f in iter_tgz(fileobj):
            rel = front_relpath(name)
            if rel is None or (wanted is not None and rel not in wanted):
                continue
            data = f.read()
            if max_width:
                data = resize_jpeg(data, max_width)
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mode = 0o444
            out.addfile(info, io.BytesIO(data))
            n += 1
            nbytes += len(data)
            logs.add(rel.split("/", 1)[0])
    tmp.replace(out_tar)
    return {"frames": n, "bytes": nbytes, "logs": len(logs)}


def build_index(tar_path: Path, index_path: Path) -> pd.DataFrame:
    with tarfile.open(tar_path, "r:") as tf:
        rows = [(m.name, m.offset_data, m.size) for m in tf if m.isfile()]
    df = pd.DataFrame(rows, columns=["f0_path", "offset", "size"])
    df["log_name"] = df["f0_path"].str.split("/").str[0]
    tmp = index_path.with_name(index_path.name + ".partial")
    df.to_parquet(tmp, index=False)
    tmp.replace(index_path)
    return df


def read_frame(tar_path: Path, offset: int, size: int) -> bytes:
    with open(tar_path, "rb") as f:
        f.seek(offset)
        return f.read(size)


def verify_shard(tar_path: Path, index_path: Path, expected_frames: int) -> list[str]:
    problems = []
    idx = pd.read_parquet(index_path)
    if len(idx) != expected_frames:
        problems.append(f"index has {len(idx)} frames, expected {expected_frames}")
    if not idx["f0_path"].is_unique:
        problems.append("duplicate frame paths")
    with open(tar_path, "rb") as f:
        for row in idx.itertuples():
            f.seek(row.offset)
            try:
                Image.open(io.BytesIO(f.read(row.size))).verify()
            except Exception as e:  # noqa: BLE001 - any decode failure is a bad frame
                problems.append(f"{row.f0_path}: {e}")
                if len(problems) > 20:
                    break
    return problems


def ingest_archive(split: str, archive: str, url: str, manifest: Manifest,
                   wanted: set[str] | None = None, max_width: int | None = None,
                   force: bool = False) -> None:
    key = f"{split}/{archive}"
    if manifest.state("frames", key) == "verified" and not force:
        log.info("frames %s already verified, skipping", key)
        return
    tar_path, index_path = shard_paths(split, archive)
    manifest.set("frames", key, "pending", src=url, max_width=max_width,
                 jpeg_quality=JPEG_QUALITY if max_width else None)
    for attempt in range(1, STREAM_ATTEMPTS + 1):
        try:
            reader = open_source(url)
            stats = extract_front(reader, tar_path, wanted=wanted, max_width=max_width)
            reader.close()
            break
        except STREAM_ERRORS as e:
            if attempt == STREAM_ATTEMPTS:
                manifest.set("frames", key, "failed", error=repr(e), attempts=attempt)
                raise
            log.warning("frames %s: stream error %r (attempt %d/%d), retrying", key, e, attempt,
                        STREAM_ATTEMPTS)
            time.sleep(RETRY_WAIT_S)
    build_index(tar_path, index_path)
    manifest.set("frames", key, "done", src_bytes=reader.nbytes, **stats)
    log.info("frames %s: %d front frames (%.2f GB) from %.2f GB streamed", key,
             stats["frames"], stats["bytes"] / 1e9, reader.nbytes / 1e9)
    problems = verify_shard(tar_path, index_path, stats["frames"])
    if problems:
        manifest.set("frames", key, "failed", problems=problems[:20])
        raise RuntimeError(f"shard {key} failed verification: {problems[:3]}")
    manifest.set("frames", key, "verified")


def adopt_shards(split: str, manifest: Manifest) -> list[str]:
    """Register shards produced elsewhere (copied without their manifest) after verifying
    every frame against the index; shards already verified here are skipped."""
    adopted = []
    for tar_path in sorted((paths.OPENSCENE_FRAMES / split).glob("*.tar")):
        archive = tar_path.stem
        key = f"{split}/{archive}"
        if manifest.state("frames", key) == "verified":
            continue
        index_path = shard_paths(split, archive)[1]
        if not index_path.exists():
            log.warning("frames %s: no index, skipped", key)
            continue
        n = len(pd.read_parquet(index_path, columns=["f0_path"]))
        problems = verify_shard(tar_path, index_path, n)
        if problems:
            manifest.set("frames", key, "failed", problems=problems[:20], adopted=True)
            log.warning("frames %s: failed verification: %s", key, problems[:3])
            continue
        manifest.set("frames", key, "verified", frames=n, bytes=tar_path.stat().st_size, adopted=True)
        adopted.append(key)
    log.info("adopted %d %s shards", len(adopted), split)
    return adopted
