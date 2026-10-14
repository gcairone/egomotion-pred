"""comma2k19 (MIT): development dataset. One 1-minute segment = 20 Hz HEVC video of the
road-facing camera plus per-frame camera poses in ECEF.

Pose convention (comma "device" frame): x forward, y right, z down. The device is mounted
on the windshield with a small, unknown pitch/yaw with respect to the car, so the device
frame is not a usable ego frame. As openpilot does, we calibrate the mount from motion: the
mean direction of travel expressed in the device frame becomes the ego x axis (z up,
y left), with the origin on the ground below the camera.
"""

import urllib.parse
from pathlib import Path

import numpy as np

from . import paths, stream
from .geometry import make_T, quat_to_rot

REVISION = "4c7f1a6e1957745beadc1def0e7225f559b09a2a"
RAW_BASE = f"https://raw.githubusercontent.com/commaai/comma2k19/{REVISION}/"
EXAMPLE_SEGMENT = "Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40"
SEGMENT_FILES = [
    "video.hevc",
    "global_pose/frame_times",
    "global_pose/frame_positions",
    "global_pose/frame_orientations",
    "global_pose/frame_velocities",
    "processed_log/CAN/speed/t",
    "processed_log/CAN/speed/value",
]

# Road camera of the comma EON (openpilot): 1164x874, focal 910 px, no distortion model.
IMAGE_SIZE = (1164, 874)
K = np.array([[910.0, 0.0, 582.0], [0.0, 910.0, 437.0], [0.0, 0.0, 1.0]])
CAMERA_HEIGHT = 1.22  # m above the road, value used by openpilot for path drawing
BONNET_ROW = 620  # image rows below this show the car's own bonnet (example segment, RAV4)
FPS = 20
MIN_CALIB_SPEED = 5.0  # m/s, frames used to estimate the mount orientation
# OpenCV camera axes (right, down, forward) in device axes (forward, right, down)
R_DEV_CAM = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def segment_dir(segment: str = EXAMPLE_SEGMENT) -> Path:
    return paths.RAW / "comma2k19" / segment.replace("|", "_")


def fetch_segment(segment: str = EXAMPLE_SEGMENT) -> Path:
    dest = segment_dir(segment)
    for f in SEGMENT_FILES:
        target = dest / f
        if not target.exists():
            stream.download(RAW_BASE + urllib.parse.quote(f"{segment}/{f}"), target)
    return dest


def mount_calibration(pos: np.ndarray, R_ecef_dev: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, dict]:
    """R_dev_ego: ego axes expressed in device axes, from the mean direction of travel."""
    vel = np.diff(pos, axis=0) / np.diff(t)[:, None]
    speed = np.linalg.norm(vel, axis=1)
    sel = speed > MIN_CALIB_SPEED
    if sel.sum() < 10:
        raise ValueError("not enough moving frames to calibrate the camera mount")
    v_dev = np.einsum("nji,nj->ni", R_ecef_dev[:-1][sel], vel[sel] / speed[sel, None])
    x = v_dev.mean(axis=0)
    x /= np.linalg.norm(x)
    up = np.array([0.0, 0.0, -1.0])  # device z points down
    y = np.cross(up, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    R_dev_ego = np.stack([x, y, z], axis=1)
    info = {
        "pitch_deg": float(np.degrees(np.arcsin(-x[2]))),  # >0: travel direction above device x
        "yaw_deg": float(np.degrees(np.arctan2(-x[1], x[0]))),  # >0: travel to the left of device x
        "frames": int(sel.sum()),
    }
    return R_dev_ego, info


def load_poses(seg: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Frame times (s), T_ecef_ego (N, 4, 4), T_ego_cam (4, 4) and the mount calibration."""
    t = np.load(seg / "global_pose/frame_times")
    pos = np.load(seg / "global_pose/frame_positions")
    R_ecef_dev = quat_to_rot(np.load(seg / "global_pose/frame_orientations"))
    R_dev_ego, calib = mount_calibration(pos, R_ecef_dev, t)
    R_ecef_ego = R_ecef_dev @ R_dev_ego
    origin = pos - CAMERA_HEIGHT * R_ecef_ego[:, :, 2]  # ground point below the camera
    T_ego_cam = make_T(R_dev_ego.T @ R_DEV_CAM, [0.0, 0.0, CAMERA_HEIGHT])
    return t, make_T(R_ecef_ego, origin), T_ego_cam, calib


def iter_frames(seg: Path, every: int = 1):
    """Yield (frame index, RGB uint8 array) for every `every`-th frame of the segment video."""
    import av

    with av.open(str(seg / "video.hevc")) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i % every == 0:
                yield i, frame.to_ndarray(format="rgb24")
