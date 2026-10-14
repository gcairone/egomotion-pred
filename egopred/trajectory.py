"""Dataset-agnostic trajectory windows, quality checks, scenario tags, B0 baselines, metrics.

A window is centred on a reference frame t0 (the "current" frame): HIST frames of history
including t0 and FUT future waypoints, all at DT spacing and expressed as [x, y, yaw] in the
ego frame at t0 (NAVSIM task format: 4 history frames, 8 waypoints over 4 s).
"""

import numpy as np

from .geometry import poses_to_xyyaw, relative_poses, wrap_angle

DT = 0.5
HIST = 4
FUT = 8
OFFSETS = np.arange(-(HIST - 1), FUT + 1)
T_FUT = DT * np.arange(1, FUT + 1)

# Quality thresholds
GAP_TOL = 0.1  # s, tolerated deviation of a frame interval from DT
MAX_SPEED = 45.0  # m/s between consecutive poses; above this is a pose jump
MAX_YAW_STEP = 1.0  # rad between consecutive poses (0.5 s)

# Scenario thresholds (on the 4 s future unless stated)
STATIC_SPEED = 0.5  # m/s
STATIC_PATH = 1.0  # m travelled
TURN_DEG = 30.0  # |heading change| above this: turn (typically at an intersection)
CURVE_DEG = 10.0  # |heading change| in (CURVE_DEG, TURN_DEG]: curve
# Lateral shift: heading back near the start but displaced sideways. From the trajectory
# alone a lane change and an S-bend of the road look the same; telling them apart needs
# the map (lane assignment), so the tag deliberately does not claim "lane change".
LATERAL_SHIFT_DEG = 8.0  # max |heading change|
LATERAL_SHIFT_LAT = 2.0  # m, min |lateral offset| at 4 s
SPEED_CHANGE = 2.0  # m/s change over 4 s for brake / accelerate

SCENARIOS = ["static", "start", "turn_left", "turn_right", "lateral_shift", "curve",
             "brake", "accel", "cruise"]


def windows(t_sec: np.ndarray, T_world_ego: np.ndarray, ref: np.ndarray) -> dict:
    """Build windows for reference indices `ref` of one sequence.

    Indices whose window does not fit inside the sequence must be filtered by the caller
    (see `valid_refs`). Returns history/future [x, y, yaw], future z and QC flags."""
    ref = np.asarray(ref, dtype=np.int64)
    T_rel = relative_poses(T_world_ego, ref, OFFSETS)
    xyyaw = poses_to_xyyaw(T_rel)
    idx = ref[:, None] + OFFSETS[None, :]
    dts = np.diff(t_sec[idx], axis=1)
    step = np.linalg.norm(np.diff(xyyaw[:, :, :2], axis=1), axis=2)
    dyaw = np.abs(wrap_angle(np.diff(xyyaw[:, :, 2], axis=1)))
    hist = xyyaw[:, :HIST]
    return {
        "hist": hist,
        "fut": xyyaw[:, HIST:],
        "fut_z": T_rel[:, HIST:, 2, 3],
        "gap": (np.abs(dts - DT) > GAP_TOL).any(axis=1),
        "pose_jump": ((step / np.maximum(dts, 1e-3)) > MAX_SPEED).any(axis=1) | (dyaw > MAX_YAW_STEP).any(axis=1),
        # kinematics from the last history step (t0 - DT -> t0)
        "speed_pose": step[:, HIST - 2] / np.maximum(dts[:, HIST - 2], 1e-3),
        "yaw_rate": wrap_angle(hist[:, -1, 2] - hist[:, -2, 2]) / np.maximum(dts[:, HIST - 2], 1e-3),
    }


def valid_refs(n: int) -> np.ndarray:
    return np.arange(HIST - 1, n - FUT)


def future_speeds(fut: np.ndarray) -> np.ndarray:
    pts = np.concatenate([np.zeros_like(fut[:, :1, :2]), fut[:, :, :2]], axis=1)
    return np.linalg.norm(np.diff(pts, axis=1), axis=2) / DT


def tag_scenarios(fut: np.ndarray, v0: np.ndarray) -> dict:
    """Multi-label scenario flags plus one primary label per sample (priority = SCENARIOS order)."""
    dpsi = np.degrees(wrap_angle(fut[:, -1, 2]))
    v_fut = future_speeds(fut)
    path = v_fut.sum(axis=1) * DT
    v_end = v_fut[:, -2:].mean(axis=1)
    lat = fut[:, -1, 1]
    static = (v0 < STATIC_SPEED) & (path < STATIC_PATH)
    moving = ~static
    flags = {
        "static": static,
        "start": moving & (v0 < STATIC_SPEED),
        "turn_left": moving & (dpsi > TURN_DEG),
        "turn_right": moving & (dpsi < -TURN_DEG),
        "lateral_shift": moving & (np.abs(dpsi) < LATERAL_SHIFT_DEG) & (np.abs(lat) > LATERAL_SHIFT_LAT),
        "curve": moving & (np.abs(dpsi) > CURVE_DEG) & (np.abs(dpsi) <= TURN_DEG),
        "brake": moving & (v0 - v_end > SPEED_CHANGE),
        "accel": moving & (v_end - v0 > SPEED_CHANGE),
    }
    flags["cruise"] = moving & ~np.any([flags[k] for k in SCENARIOS[1:-1]], axis=0)
    primary = np.full(len(fut), "", dtype=object)
    for name in reversed(SCENARIOS):
        primary[flags[name]] = name
    return {
        "flags": flags,
        "primary": primary,
        "heading_change_deg": dpsi,
        "path_length": path,
        "v_end": v_end,
        "lateral_end": lat,
    }


def constant_velocity(v_ego: np.ndarray) -> np.ndarray:
    """B0-CV: keep the current ego-frame velocity vector (M, 2). Returns (M, FUT, 3)."""
    xy = v_ego[:, None, :] * T_FUT[None, :, None]
    yaw = np.zeros(xy.shape[:2] + (1,))
    return np.concatenate([xy, yaw], axis=2)


def constant_turn_rate(speed: np.ndarray, yaw_rate: np.ndarray) -> np.ndarray:
    """B0-CTRV: constant speed and yaw rate (constant curvature arc). Returns (M, FUT, 3)."""
    s = speed[:, None]
    w = yaw_rate[:, None]
    t = T_FUT[None, :]
    yaw = w * t
    small = np.abs(w) < 1e-4
    w_safe = np.where(small, 1.0, w)
    x = np.where(small, s * t, s * np.sin(yaw) / w_safe)
    y = np.where(small, 0.5 * s * w * t**2, s * (1 - np.cos(yaw)) / w_safe)
    return np.stack([x, y, yaw], axis=2)


def displacement_metrics(pred: np.ndarray, gt: np.ndarray, miss_threshold: float = 2.0) -> dict:
    """Per-sample errors of a single-hypothesis prediction. pred/gt: (M, FUT, >=2)."""
    d = pred[:, :, :2] - gt[:, :, :2]
    err = np.linalg.norm(d, axis=2)
    tangent = np.stack([np.cos(gt[:, :, 2]), np.sin(gt[:, :, 2])], axis=2)
    normal = np.stack([-tangent[..., 1], tangent[..., 0]], axis=2)
    return {
        "ade": err.mean(axis=1),
        "fde": err[:, -1],
        "miss": err[:, -1] > miss_threshold,
        "lat": np.abs((d * normal).sum(axis=2)).mean(axis=1),
        "lon": np.abs((d * tangent).sum(axis=2)).mean(axis=1),
    }
