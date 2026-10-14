"""Rigid transforms, ego-frame trajectories and camera projection.

Conventions used throughout the project:
- ego frame: x forward, y left, z up (right-handed); yaw is counter-clockwise about z.
- camera frame: OpenCV, x right, y down, z forward.
- poses are 4x4 homogeneous matrices T_a_b mapping points from frame b to frame a.
"""

import numpy as np

# Rotation taking OpenCV camera axes to ego axes for a camera looking straight ahead.
R_EGO_CAM_FORWARD = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def quat_to_rot(q) -> np.ndarray:
    """Quaternion(s) [w, x, y, z] -> rotation matrix (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def make_T(R, t) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def inv_T(T) -> np.ndarray:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    return make_T(Rt, -np.einsum("...ij,...j->...i", Rt, t))


def yaw_of(T) -> np.ndarray:
    """Heading (rotation about z) of pose(s) T."""
    return np.arctan2(T[..., 1, 0], T[..., 0, 0])


def step_yaw(T, span: int = 1) -> np.ndarray:
    """Yaw change from pose i to pose i + span, about pose i's own z axis (valid in any
    world frame, including ECEF where the world z axis is not the local vertical)."""
    R = T[:, :3, :3]
    R_rel = np.einsum("nji,njk->nik", R[:-span], R[span:])
    return np.arctan2(R_rel[:, 1, 0], R_rel[:, 0, 0])


def wrap_angle(a) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def relative_poses(T_world_ego: np.ndarray, ref: np.ndarray, offsets) -> np.ndarray:
    """Poses at `ref + offsets` expressed in the ego frame at `ref`.

    T_world_ego: (N, 4, 4) poses of one sequence; ref: (M,) indices; offsets: (K,) ints.
    Returns (M, K, 4, 4). Caller guarantees all indices are in range."""
    ref = np.asarray(ref)
    idx = ref[:, None] + np.asarray(offsets)[None, :]
    T0_inv = inv_T(T_world_ego[ref])
    return np.einsum("mij,mkjl->mkil", T0_inv, T_world_ego[idx])


def poses_to_xyyaw(T_rel: np.ndarray) -> np.ndarray:
    """(..., 4, 4) relative poses -> (..., 3) [x, y, yaw]."""
    return np.concatenate([T_rel[..., :2, 3], yaw_of(T_rel)[..., None]], axis=-1)


def distort(xn: np.ndarray, yn: np.ndarray, dist) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply OpenCV distortion (k1, k2, p1, p2, k3) to normalized coordinates.

    Also returns a validity mask: strong barrel distortion folds points far outside the
    field of view back into the image, so points where the radial map stops being
    monotonic are flagged invalid."""
    k1, k2, p1, p2, k3 = (list(np.asarray(dist, dtype=np.float64).ravel()) + [0.0] * 5)[:5]
    r2 = xn * xn + yn * yn
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    monotonic = 1 + 3 * k1 * r2 + 5 * k2 * r2**2 + 7 * k3 * r2**3 > 0
    return xd, yd, monotonic


def project(points_cam: np.ndarray, K: np.ndarray, dist=None, min_depth: float = 0.5):
    """Project (N, 3) camera-frame points to pixels. Returns (uv (N, 2), valid (N,))."""
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    in_front = Z > min_depth
    Zs = np.where(in_front, Z, 1.0)
    xn, yn = X / Zs, Y / Zs
    if dist is not None and np.any(np.asarray(dist) != 0):
        xn, yn, ok = distort(xn, yn, dist)
    else:
        ok = np.ones_like(in_front)
    u = K[0, 0] * xn + K[0, 1] * yn + K[0, 2]
    v = K[1, 1] * yn + K[1, 2]
    return np.stack([u, v], axis=1), in_front & ok


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]
