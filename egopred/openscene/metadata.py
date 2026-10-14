"""OpenScene metadata: stream the per-log pickles and keep a compact per-frame table.

Each log pickle holds a list of per-frame dicts (2 Hz) with poses, kinematics, driving
command, calibration of all cameras and object annotations. We keep what the project
needs (front camera only, no annotations) as one parquet file per log.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import paths, safe_pickle
from ..manifest import Manifest
from ..stream import iter_tgz, open_source
from .sources import metadata_url

log = logging.getLogger(__name__)

LOG_NAME_RE = re.compile(r"^[\w.\-]+$")
STR_COLS = [
    "token", "log_name", "log_token", "scene_name", "scene_token", "map_location",
    "vehicle_name", "sample_prev", "sample_next", "f0_path",
]
# prefix -> number of flattened float values
ARRAY_COLS = {
    "e2g": 16,  # ego2global, row-major 4x4
    "l2e": 16,  # lidar2ego
    "dyn": 4,  # ego_dynamic_state: vx, vy, ax, ay (ego frame)
    "can": 18,  # can_bus
    "f0K": 9,  # CAM_F0 intrinsics
    "f0dist": 5,  # CAM_F0 distortion k1, k2, p1, p2, k3
    "f0R": 9,  # CAM_F0 sensor2lidar rotation
    "f0t": 3,  # CAM_F0 sensor2lidar translation
}


def cols(prefix: str) -> list[str]:
    return [f"{prefix}_{i}" for i in range(ARRAY_COLS[prefix])]


def array(df: pd.DataFrame, prefix: str, shape=None) -> np.ndarray:
    a = df[cols(prefix)].to_numpy(dtype=np.float64)
    return a if shape is None else a.reshape((len(df),) + tuple(shape))


def _vec(value, n: int) -> np.ndarray:
    if value is None:
        return np.full(n, np.nan)
    a = np.asarray(value, dtype=np.float64).ravel()
    return a if a.size == n else np.full(n, np.nan)


def compact_log(records: list[dict]) -> pd.DataFrame:
    """Per-frame dicts of one log -> flat DataFrame sorted by timestamp."""
    rows = []
    for r in records:
        cam = (r.get("cams") or {}).get("CAM_F0") or {}
        row = {k: r.get(k) for k in STR_COLS if k != "f0_path"}
        row["f0_path"] = cam.get("data_path")
        row["frame_idx"] = int(r.get("frame_idx", -1))
        row["timestamp"] = int(r["timestamp"])
        cmd = _vec(r.get("driving_command"), 4)
        row.update({f"cmd_{i}": (int(c) if np.isfinite(c) else -1) for i, c in enumerate(cmd)})
        anns = r.get("anns") or {}
        row["n_boxes"] = len(anns.get("gt_names", []))
        vals = {
            "e2g": r.get("ego2global"),
            "l2e": r.get("lidar2ego"),
            "dyn": r.get("ego_dynamic_state"),
            "can": r.get("can_bus"),
            "f0K": cam.get("cam_intrinsic"),
            "f0dist": cam.get("distortion"),
            "f0R": cam.get("sensor2lidar_rotation"),
            "f0t": cam.get("sensor2lidar_translation"),
        }
        for prefix, v in vals.items():
            row.update(zip(cols(prefix), _vec(v, ARRAY_COLS[prefix])))
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in STR_COLS:
        df[c] = df[c].astype("string")
    df = df.astype({f"cmd_{i}": "int8" for i in range(4)} | {"frame_idx": "int32", "n_boxes": "int32"})
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _write_parquet(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
    tmp.replace(dest)


def ingest_metadata(split: str, manifest: Manifest, src: str | None = None, force: bool = False) -> None:
    """Stream openscene_metadata_<split>.tgz and write data/openscene/meta/<split>/<log>.parquet."""
    if manifest.state("meta_archive", split) == "verified" and not force:
        log.info("metadata %s already verified, skipping", split)
        return
    out_dir = paths.OPENSCENE_META / split
    kind = f"meta_log:{split}"
    n_new = n_skip = 0
    reader = open_source(src or metadata_url(split))
    manifest.set("meta_archive", split, "pending", src=src or metadata_url(split))
    for name, _size, f in iter_tgz(reader):
        if not name.endswith(".pkl"):
            continue
        log_name = Path(name).stem
        if not LOG_NAME_RE.match(log_name):
            raise ValueError(f"unexpected log name in archive: {name!r}")
        dest = out_dir / f"{log_name}.parquet"
        if not force and manifest.state(kind, log_name) in ("done", "verified") and dest.exists():
            n_skip += 1
            continue
        df = compact_log(safe_pickle.loads(f.read()))
        _write_parquet(df, dest)
        manifest.set(kind, log_name, "done", rows=len(df))
        n_new += 1
        if n_new % 100 == 0:
            log.info("metadata %s: %d logs written", split, n_new)
    reader.close()
    manifest.set("meta_archive", split, "done", logs=n_new + n_skip, src_bytes=reader.nbytes)
    log.info("metadata %s: %d logs written, %d already present", split, n_new, n_skip)
    verify_metadata(split, manifest)


def check_log(df: pd.DataFrame) -> list[str]:
    """Structural problems of one compact log table (empty list = OK)."""
    problems = []
    if df["token"].isna().any() or not df["token"].is_unique:
        problems.append("missing or duplicate tokens")
    if (np.diff(df["timestamp"].to_numpy()) <= 0).any():
        problems.append("non-increasing timestamps")
    if not np.isfinite(array(df, "e2g")).all():
        problems.append("non-finite ego2global")
    if df["f0_path"].isna().any():
        problems.append("missing CAM_F0 path")
    if df["log_name"].nunique() != 1:
        problems.append("more than one log_name")
    return problems


def verify_metadata(split: str, manifest: Manifest) -> bool:
    kind = f"meta_log:{split}"
    bad = []
    tokens = 0
    seen: set[str] = set()
    for log_name in manifest.keys(kind):
        state, info = manifest.get(kind, log_name)
        path = paths.OPENSCENE_META / split / f"{log_name}.parquet"
        if not path.exists():
            bad.append((log_name, "file missing"))
            continue
        df = pd.read_parquet(path)
        problems = check_log(df)
        if len(df) != info.get("rows"):
            problems.append(f"rows {len(df)} != recorded {info.get('rows')}")
        dup = seen.intersection(df["token"].tolist())
        if dup:
            problems.append(f"{len(dup)} tokens also in other logs")
        seen.update(df["token"].tolist())
        tokens += len(df)
        if problems:
            bad.append((log_name, "; ".join(problems)))
            manifest.set(kind, log_name, "failed", problems=problems)
        elif state != "verified":
            manifest.set(kind, log_name, "verified")
    if bad:
        for name, why in bad[:20]:
            log.error("metadata %s/%s: %s", split, name, why)
        manifest.set("meta_archive", split, "failed", bad_logs=len(bad))
        return False
    manifest.set("meta_archive", split, "verified", frames=tokens)
    log.info("metadata %s verified: %d logs, %d frames", split, len(manifest.keys(kind)), tokens)
    return True


def log_files(split: str) -> list[Path]:
    return sorted((paths.OPENSCENE_META / split).glob("*.parquet"))


def load_log(split: str, log_name: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(paths.OPENSCENE_META / split / f"{log_name}.parquet", columns=columns)
