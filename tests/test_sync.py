import cv2
import numpy as np

from egopred.geometry import R_EGO_CAM_FORWARD, inv_T, make_T, project, transform_points
from egopred.sync import camera_rotation, ego_yaw, lag_correlation, sync_summary, track

K = np.array([[800.0, 0, 480], [0, 800.0, 270], [0, 0, 1]])
DIST = [-0.35, 0.17, -0.002, 0.0005, -0.05]


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_rotation_from_correspondences_with_distortion():
    rng = np.random.default_rng(0)
    T_ego_cam = make_T(R_EGO_CAM_FORWARD, [1.5, 0, 1.5])
    pts = np.c_[rng.uniform(8, 60, 400), rng.uniform(-15, 15, 400), rng.uniform(-1, 6, 400)]
    yaw = np.radians(4.0)
    T_a = make_T(np.eye(3), [0, 0, 0])
    T_b = make_T(rot_z(yaw), [4.0, 0.3, 0])  # moved forward and turned left
    uv = []
    for T in (T_a, T_b):
        cam = transform_points(inv_T(T @ T_ego_cam), pts)
        p, ok = project(cam, K, DIST)
        uv.append((p, ok))
    ok = uv[0][1] & uv[1][1] & (uv[0][0][:, 0] > 0) & (uv[0][0][:, 0] < 960)
    R, n = camera_rotation(uv[0][0][ok], uv[1][0][ok], K, DIST)
    assert n > 100
    assert abs(ego_yaw(R, R_EGO_CAM_FORWARD) - yaw) < np.radians(0.05)


def test_track_follows_image_shift():
    rng = np.random.default_rng(1)
    a = cv2.GaussianBlur(rng.integers(0, 255, (240, 320), dtype=np.uint8), (5, 5), 0)
    b = np.roll(a, (2, 6), axis=(0, 1))
    p0, p1 = track(a, b)
    keep = (p0[:, 0] > 20) & (p0[:, 0] < 290) & (p0[:, 1] > 20) & (p0[:, 1] < 210)
    assert keep.sum() > 50
    np.testing.assert_allclose(np.median(p1[keep] - p0[keep], axis=0), [6, 2], atol=0.1)


def test_lag_correlation_and_verdicts():
    rng = np.random.default_rng(2)
    pose = np.convolve(rng.normal(size=300), np.ones(5) / 5, mode="same") * 0.1
    img_lead = np.roll(pose, -1)  # image associated one frame late
    corr = lag_correlation(img_lead, pose, range(-2, 3))
    assert max(corr, key=corr.get) == 1
    s = sync_summary(pose + 0.002 * rng.normal(size=300), pose, range(-2, 3))
    assert s["best_lag"] == 0 and s["verdict"] == "pass" and abs(s["gain"] - 1) < 0.05
    assert sync_summary(img_lead, pose, range(-2, 3))["verdict"] == "fail"
    assert sync_summary(pose * 0.01, pose * 0.01, range(-2, 3))["verdict"].startswith("inconclusive")
    nan_img = pose.copy()
    nan_img[::7] = np.nan
    assert sync_summary(nan_img, pose, range(-2, 3))["verdict"] == "pass"
