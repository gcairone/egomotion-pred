"""NAVSIM sample tables: one row per (scene filter token = current frame), built log by log.

Sample id = OpenScene token of the current frame, so features (keyed by frame token),
targets and views stay aligned. Rows store the 4 history frame tokens / image paths, the
ego-frame history and 8 future waypoints, kinematics, driving command, QC flags,
scenario tags and the drive-level, city-stratified train/val assignment.
"""

import hashlib
import logging

import numpy as np
import pandas as pd
import yaml

from .. import paths, stream
from ..trajectory import FUT, HIST, OFFSETS, tag_scenarios, valid_refs, windows
from .metadata import array, cols, load_log, log_files
from .sources import SCENE_FILTER_URL

log = logging.getLogger(__name__)

VAL_PERCENT = 10
COLUMNS = ["token", "log_name", "timestamp", "map_location", "f0_path", "n_boxes",
           *cols("e2g"), *cols("dyn"), *(f"cmd_{i}" for i in range(4))]


def drive_id(log_name: str) -> str:
    """OpenScene logs are segments of longer nuPlan drives: '<date>_<vehicle>_<start>_<end>'.
    Splitting by drive keeps segments of the same drive on the same side."""
    return "_".join(log_name.split("_")[:2])


def _drive_hash(drive: str) -> int:
    return int(hashlib.md5(drive.encode()).hexdigest(), 16)


def assign_val_drives(drives: pd.Series, cities: pd.Series) -> set[str]:
    """Whole drives for validation, ~VAL_PERCENT of the samples of each city.

    Drives are few and long, so a plain hash threshold gives a city mix far from the
    training one; instead, per city, drives are taken in a fixed pseudo-random order
    (md5 of the drive id) until the city's share is reached."""
    val: set[str] = set()
    counts = pd.DataFrame({"drive": drives, "city": cities}).groupby(["city", "drive"]).size()
    for city, per_drive in counts.groupby(level=0):
        per_drive = per_drive.droplevel(0)
        target = VAL_PERCENT / 100 * per_drive.sum()
        taken = 0
        for d in sorted(per_drive.index, key=_drive_hash):
            if taken >= target:
                break
            val.add(d)
            taken += per_drive[d]
    return val


def scene_filter_tokens(name: str) -> list[str]:
    dest = paths.SCENE_FILTERS / f"{name}.yaml"
    if not dest.exists():
        stream.download(SCENE_FILTER_URL.format(name=name), dest)
    cfg = yaml.safe_load(dest.read_text())
    assert cfg["num_history_frames"] == HIST, cfg["num_history_frames"]
    return list(cfg["tokens"])


def _flat(prefix: str, a: np.ndarray, names=("x", "y", "yaw")) -> dict:
    out = {}
    for k in range(a.shape[1]):
        if a.ndim == 2:
            out[f"{prefix}_{k}"] = a[:, k]
        else:
            for j, n in enumerate(names):
                out[f"{prefix}_{n}_{k}"] = a[:, k, j]
    return out


def unflat(df: pd.DataFrame, prefix: str, n: int, names=("x", "y", "yaw")) -> np.ndarray:
    return np.stack([df[[f"{prefix}_{c}_{k}" for c in names]].to_numpy() for k in range(n)], axis=1)


def log_samples(df: pd.DataFrame, tokens: set[str] | None) -> pd.DataFrame | None:
    """Samples of one log. tokens=None: every frame with a complete window."""
    refs = valid_refs(len(df))
    if tokens is not None:
        tok = df["token"].to_numpy(dtype=object)[refs]
        refs = refs[np.fromiter((t in tokens for t in tok), dtype=bool, count=len(tok))]
    if len(refs) == 0:
        return None
    t = df["timestamp"].to_numpy() / 1e6
    T = array(df, "e2g", (4, 4))
    w = windows(t, T, refs)
    dyn = array(df, "dyn")[refs]
    v0 = np.hypot(dyn[:, 0], dyn[:, 1])
    tags = tag_scenarios(w["fut"], v0)
    cmd = df[[f"cmd_{i}" for i in range(4)]].to_numpy()[refs]
    idx = refs[:, None] + OFFSETS[None, :HIST]
    out = {
        "sample_id": df["token"].to_numpy(dtype=object)[refs],
        "log_name": df["log_name"].to_numpy(dtype=object)[refs],
        "timestamp": df["timestamp"].to_numpy()[refs],
        "map_location": df["map_location"].to_numpy(dtype=object)[refs],
        "n_boxes": df["n_boxes"].to_numpy()[refs],
        **{f"hist_token_{k}": df["token"].to_numpy(dtype=object)[idx[:, k]] for k in range(HIST)},
        **{f"hist_f0_{k}": df["f0_path"].to_numpy(dtype=object)[idx[:, k]] for k in range(HIST)},
        **_flat("hist", w["hist"]),
        **_flat("fut", w["fut"]),
        **_flat("fut_z", w["fut_z"]),
        "vx": dyn[:, 0], "vy": dyn[:, 1], "ax": dyn[:, 2], "ay": dyn[:, 3],
        "speed": v0,
        "speed_pose": w["speed_pose"],
        "yaw_rate": w["yaw_rate"],
        "command": np.where(cmd.max(axis=1) > 0, cmd.argmax(axis=1), -1).astype(np.int8),
        "gap": w["gap"],
        "pose_jump": w["pose_jump"],
        "scenario": tags["primary"],
        "heading_change_deg": tags["heading_change_deg"],
        "path_length": tags["path_length"],
        "v_end": tags["v_end"],
        "lateral_end": tags["lateral_end"],
        **{f"is_{k}": v for k, v in tags["flags"].items()},
    }
    return pd.DataFrame(out)


def build_samples(split: str, name: str, tokens: list[str] | None) -> pd.DataFrame:
    """Build and save data/samples/<name>.parquet from the metadata of `split`."""
    wanted = None if tokens is None else set(tokens)
    parts = []
    for f in log_files(split):
        df = load_log(split, f.stem, columns=COLUMNS)
        s = log_samples(df, wanted)
        if s is not None:
            parts.append(s)
    out = pd.concat(parts, ignore_index=True)
    out["drive_id"] = out["log_name"].map(drive_id)
    val = assign_val_drives(out["drive_id"], out["map_location"])
    out["subset"] = np.where(out["drive_id"].isin(val), "val", "train")
    if wanted is not None:
        missing = len(wanted) - out["sample_id"].nunique()
        log.info("%s: %d/%d filter tokens found (%d missing)", name, len(out), len(wanted), missing)
    paths.SAMPLES.mkdir(parents=True, exist_ok=True)
    dest = paths.SAMPLES / f"{name}.parquet"
    tmp = dest.with_name(dest.name + ".partial")
    out.to_parquet(tmp, index=False)
    tmp.replace(dest)
    log.info("wrote %s: %d samples", dest, len(out))
    return out


def load_samples(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(paths.SAMPLES / f"{name}.parquet", columns=columns)


def future(df: pd.DataFrame) -> np.ndarray:
    return unflat(df, "fut", FUT)


def history(df: pd.DataFrame) -> np.ndarray:
    return unflat(df, "hist", HIST)
