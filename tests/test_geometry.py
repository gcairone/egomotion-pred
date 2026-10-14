import numpy as np

from egopred.geometry import (R_EGO_CAM_FORWARD, inv_T, make_T, poses_to_xyyaw, project, quat_to_rot,
                              relative_poses, step_yaw, transform_points, wrap_angle, yaw_of)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_quat_to_rot_matches_axis_angle():
    a = 0.7
    q = [np.cos(a / 2), 0, 0, np.sin(a / 2)]
    np.testing.assert_allclose(quat_to_rot(q), rot_z(a), atol=1e-12)


def test_inverse_roundtrip():
    T = make_T(quat_to_rot([0.9, 0.1, -0.3, 0.2]), [5.0, -2.0, 1.0])
    np.testing.assert_allclose(T @ inv_T(T), np.eye(4), atol=1e-12)


def test_relative_poses_on_circle():
    # vehicle driving a left circle of radius 20 m at constant yaw rate, arbitrary world offset
    r, n, dpsi = 20.0, 30, 0.05
    psi = np.arange(n) * dpsi + 1.0
    T = make_T(np.stack([rot_z(p) for p in psi]),
               np.c_[r * np.sin(psi), -r * np.cos(psi), np.full(n, 3.0)] + [1000.0, 500.0, 0.0])
    rel = poses_to_xyyaw(relative_poses(T, np.array([5]), np.arange(1, 4)))[0]
    k = np.arange(1, 4)
    np.testing.assert_allclose(rel[:, 2], k * dpsi, atol=1e-9)
    np.testing.assert_allclose(rel[:, 0], r * np.sin(k * dpsi), atol=1e-9)
    np.testing.assert_allclose(rel[:, 1], r * (1 - np.cos(k * dpsi)), atol=1e-9)  # left is +y


def test_yaw_and_wrap():
    assert np.isclose(yaw_of(make_T(rot_z(3.0), [0, 0, 0])), 3.0)
    assert np.isclose(wrap_angle(2 * np.pi + 0.1), 0.1)
    assert np.isclose(wrap_angle(-np.pi - 0.1), np.pi - 0.1)


def test_project_forward_point_hits_principal_point():
    K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
    T_ego_cam = make_T(R_EGO_CAM_FORWARD, [1.5, 0.0, 1.5])
    pts_ego = np.array([[50.0, 0.0, 1.5], [10.0, 2.0, 1.5], [10.0, 0.0, 0.0], [-5.0, 0, 1.5]])
    uv, ok = project(transform_points(inv_T(T_ego_cam), pts_ego), K)
    np.testing.assert_allclose(uv[0], [640, 360])
    assert uv[1, 0] < 640  # point on the left (+y) appears left in the image
    assert uv[2, 1] > 360  # point on the ground appears below the horizon
    assert ok.tolist() == [True, True, True, False]  # behind the camera


def test_distortion_matches_opencv_formula_and_flags_fold_over():
    K = np.array([[1545.0, 0, 960], [0, 1545.0, 560], [0, 0, 1]])
    dist = [-0.356, 0.1725, -0.0021, 0.0005, -0.0523]
    xn, yn = 0.3, -0.2
    uv, ok = project(np.array([[xn, yn, 1.0]]), K, dist)
    r2 = xn**2 + yn**2
    rad = 1 - 0.356 * r2 + 0.1725 * r2**2 - 0.0523 * r2**3
    xd = xn * rad + 2 * -0.0021 * xn * yn + 0.0005 * (r2 + 2 * xn**2)
    yd = yn * rad + -0.0021 * (r2 + 2 * yn**2) + 2 * 0.0005 * xn * yn
    np.testing.assert_allclose(uv[0], [1545 * xd + 960, 1545 * yd + 560])
    assert ok[0]
    _, ok_far = project(np.array([[3.0, 0.0, 1.0]]), K, dist)  # far outside the FOV
    assert not ok_far[0]


def test_step_yaw_is_frame_independent():
    psi = np.array([0.0, 0.1, 0.25, 0.2])
    T = make_T(np.stack([rot_z(p) for p in psi]), np.zeros((4, 3)))
    np.testing.assert_allclose(step_yaw(T), np.diff(psi), atol=1e-12)
    # same motion seen from a world frame whose z axis is tilted (as in ECEF)
    W = make_T(quat_to_rot([0.8, 0.3, -0.4, 0.2]), [1e6, 2e6, 3e6])
    np.testing.assert_allclose(step_yaw(W @ T), np.diff(psi), atol=1e-12)
    assert not np.allclose(np.diff(yaw_of(W @ T)), np.diff(psi), atol=1e-3)
