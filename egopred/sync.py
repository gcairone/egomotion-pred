"""Video/pose synchronization check by visual rotation.

For frame pairs (i, i + span) the camera rotation is estimated from the images alone
(feature tracking + essential matrix with RANSAC, which separates rotation from translation
and rejects moving objects), converted to an ego-frame yaw change and correlated with the
yaw change from the poses at several frame lags. With correct sync the correlation peaks at
lag 0 and the image/pose gain is ~1 (a wrong focal length or axis convention breaks the gain).
"""

import cv2
import numpy as np

MIN_INLIERS = 30


def track(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None, max_corners: int = 1000,
          fb_tol: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Corners of uint8 image `a` tracked into `b` (pyramidal LK, forward-backward checked)."""
    p0 = cv2.goodFeaturesToTrack(a, maxCorners=max_corners, qualityLevel=0.01, minDistance=7, mask=mask)
    if p0 is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    lk = dict(winSize=(21, 21), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, **lk)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, **lk)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p0 - p0r, axis=2).ravel() < fb_tol)
    return p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]


def camera_rotation(p0: np.ndarray, p1: np.ndarray, K: np.ndarray, dist=None) -> tuple[np.ndarray | None, int]:
    """Rotation R_a_b of camera b expressed in camera a, from pixel correspondences."""
    if len(p0) < MIN_INLIERS:
        return None, 0
    d = None if dist is None else np.asarray(dist, dtype=np.float64)
    n0 = cv2.undistortPoints(p0.reshape(-1, 1, 2).astype(np.float64), K, d).reshape(-1, 2)
    n1 = cv2.undistortPoints(p1.reshape(-1, 1, 2).astype(np.float64), K, d).reshape(-1, 2)
    E, inl = cv2.findEssentialMat(n0, n1, np.eye(3), method=cv2.RANSAC, prob=0.999,
                                  threshold=1.0 / K[0, 0])
    if E is None or E.shape != (3, 3):
        return None, 0
    n_inl, R, _t, _ = cv2.recoverPose(E, n0, n1, np.eye(3), mask=inl)
    if n_inl < MIN_INLIERS:
        return None, int(n_inl)
    return R.T, int(n_inl)  # recoverPose gives x_b = R x_a + t


def ego_yaw(R_cam_a_b: np.ndarray, R_ego_cam: np.ndarray) -> float:
    M = R_ego_cam @ R_cam_a_b @ R_ego_cam.T
    return float(np.arctan2(M[1, 0], M[0, 0]))


def visual_yaw(a: np.ndarray, b: np.ndarray, K: np.ndarray, dist, R_ego_cam: np.ndarray,
               mask: np.ndarray | None = None) -> tuple[float, int]:
    """Ego-frame yaw change from image a to image b (nan if not enough inliers)."""
    R, n = camera_rotation(*track(a, b, mask), K, dist)
    return (np.nan, n) if R is None else (ego_yaw(R, R_ego_cam), n)


def lag_correlation(img: np.ndarray, pose: np.ndarray, lags) -> dict[int, float]:
    """Pearson correlation of img[i] with pose[i + lag] over finite pairs."""
    n = len(img)
    out = {}
    for lag in lags:
        i = np.arange(max(0, -lag), min(n, n - lag))
        a, b = img[i], pose[i + lag]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        out[int(lag)] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")
    return out


def sync_summary(img: np.ndarray, pose: np.ndarray, lags, min_signal_rad: float = 0.01,
                 min_corr: float = 0.8, lag_tol: int = 0) -> dict:
    """img/pose: yaw change per pair, same pairing (lag 0 = as associated by the dataset)."""
    corr = lag_correlation(img, pose, lags)
    finite = {k: v for k, v in corr.items() if np.isfinite(v)}
    best = max(finite, key=finite.get) if finite else None
    ok = np.isfinite(img) & np.isfinite(pose)
    p, q = pose[ok], img[ok]
    gain = float((p * q).sum() / max((p * p).sum(), 1e-12))
    resid = float(np.sqrt(np.mean((q - p) ** 2))) if ok.any() else float("nan")
    signal = float(p.std()) if ok.any() else 0.0
    if signal < min_signal_rad:
        verdict = "inconclusive (not enough rotation)"
    elif best is not None and abs(best) <= lag_tol and corr[0] > min_corr and 0.8 < gain < 1.2:
        verdict = "pass"
    else:
        verdict = "fail"
    return {"corr_by_lag": corr, "best_lag": best, "gain": gain, "rms_diff_deg": float(np.degrees(resid)),
            "pose_yaw_std_deg": float(np.degrees(signal)), "pairs": int(ok.sum()), "verdict": verdict}
