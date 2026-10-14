import numpy as np

from egopred import comma2k19
from egopred.geometry import inv_T, project, quat_to_rot, transform_points
from egopred.trajectory import windows


def rot_to_quat(R):
    w = np.sqrt(1 + np.trace(R)) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def write_segment(tmp_path, R, n=20):
    """Vehicle driving along ECEF +x at 20 m/s; device orientation R (ECEF from device)."""
    t = np.arange(n) * 0.5
    pos = np.c_[20.0 * t, np.zeros(n), np.zeros(n)] + [-2.7e6, -4.2e6, 3.9e6]
    gp = tmp_path / "global_pose"
    gp.mkdir()
    for name, arr in [("frame_times", t), ("frame_positions", pos),
                      ("frame_orientations", np.tile(rot_to_quat(R), (n, 1)))]:
        with open(gp / name, "wb") as f:
            np.save(f, arr)
    return pos


def test_device_poses_map_to_ego_frame(tmp_path):
    # device frame (fwd, right, down) aligned with ECEF (+x, +y, +z)
    R = np.eye(3)
    pos = write_segment(tmp_path, R)
    tt, T, T_ego_cam, calib = comma2k19.load_poses(tmp_path)
    assert abs(calib["pitch_deg"]) < 1e-9 and abs(calib["yaw_deg"]) < 1e-9
    np.testing.assert_allclose(quat_to_rot(rot_to_quat(R)), R, atol=1e-12)
    w = windows(tt, T, np.array([3]))
    np.testing.assert_allclose(w["fut"][0, :, 0], 10.0 * np.arange(1, 9), atol=1e-6)  # forward
    np.testing.assert_allclose(w["fut"][0, :, 1:], 0.0, atol=1e-6)
    # ego origin is on the ground: CAMERA_HEIGHT below the device, i.e. +z (down) in device frame
    np.testing.assert_allclose(T[0, :3, 3] - pos[0], [0, 0, comma2k19.CAMERA_HEIGHT])
    # a ground point 20 m ahead projects on the image centre column, below the principal point
    uv, ok = project(transform_points(inv_T(T_ego_cam), np.array([[20.0, 0, 0]])), comma2k19.K)
    assert ok[0] and abs(uv[0, 0] - 582) < 1e-6 and uv[0, 1] > 437


def test_mount_pitch_is_calibrated_from_motion(tmp_path):
    # device pitched 3 deg nose-down: its x axis points below the direction of travel
    pitch = np.radians(3.0)
    R = rot_y(-pitch)  # rotates device x towards ECEF +z, which is "down" in this setup
    write_segment(tmp_path, R)
    tt, T, T_ego_cam, calib = comma2k19.load_poses(tmp_path)
    assert abs(calib["pitch_deg"] - 3.0) < 1e-6
    w = windows(tt, T, np.array([3]))
    np.testing.assert_allclose(w["fut"][0, :, 0], 10.0 * np.arange(1, 9), atol=1e-6)
    np.testing.assert_allclose(w["fut_z"][0], 0.0, atol=1e-6)  # travel stays level in ego frame
    # the direction of travel projects 3 deg above the principal point
    uv, _ = project(transform_points(inv_T(T_ego_cam), np.array([[1e4, 0, comma2k19.CAMERA_HEIGHT]])),
                    comma2k19.K)
    np.testing.assert_allclose(uv[0], [582, 437 - 910 * np.tan(pitch)], atol=1e-3)
