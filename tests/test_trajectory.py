import numpy as np
import pytest

from egopred.geometry import make_T
from egopred.trajectory import (FUT, HIST, constant_turn_rate, constant_velocity, displacement_metrics,
                                tag_scenarios, valid_refs, windows)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def sequence(speed, yaw_rate, n=40, dt=0.5, accel=0.0, psi0=0.4):
    """Integrate a unicycle with constant yaw rate and acceleration; returns t, T_world_ego."""
    t = np.arange(n) * dt
    x = y = 0.0
    psi, v = psi0, speed
    poses = []
    for _ in range(n):
        poses.append(make_T(rot_z(psi), [x + 100.0, y - 50.0, 2.0]))
        sub = 100
        for _ in range(sub):
            x += v * np.cos(psi) * dt / sub
            y += v * np.sin(psi) * dt / sub
            psi += yaw_rate * dt / sub
            v = max(0.0, v + accel * dt / sub)
    return t, np.stack(poses)


def test_windows_shapes_and_kinematics():
    t, T = sequence(10.0, 0.1)
    refs = valid_refs(len(t))
    assert refs[0] == HIST - 1 and refs[-1] == len(t) - FUT - 1
    w = windows(t, T, refs)
    assert w["hist"].shape == (len(refs), HIST, 3) and w["fut"].shape == (len(refs), FUT, 3)
    np.testing.assert_allclose(w["hist"][:, -1], 0.0, atol=1e-9)  # current pose is the origin
    np.testing.assert_allclose(w["yaw_rate"], 0.1, atol=1e-6)
    np.testing.assert_allclose(w["speed_pose"], 10.0, rtol=1e-3)  # chord ~ arc for small turns
    assert not w["gap"].any() and not w["pose_jump"].any()


def test_gap_and_pose_jump_flags():
    t, T = sequence(10.0, 0.0)
    t = t.copy()
    t[20:] += 0.5  # a dropped frame
    T = T.copy()
    T[30, 0, 3] += 50.0  # teleport
    w = windows(t, T, valid_refs(len(t)))
    refs = valid_refs(len(t))
    has_20 = (refs - HIST + 1 <= 19) & (refs + FUT >= 20)
    np.testing.assert_array_equal(w["gap"], has_20)
    assert w["pose_jump"][(refs + FUT >= 30) & (refs - HIST + 1 <= 30)].all()


def test_ctrv_is_exact_on_constant_turn_and_cv_on_straight():
    for yaw_rate in (0.0, 0.15, -0.2):
        t, T = sequence(8.0, yaw_rate)
        w = windows(t, T, valid_refs(len(t)))
        pred = constant_turn_rate(np.full(len(w["fut"]), 8.0), w["yaw_rate"])
        assert displacement_metrics(pred, w["fut"])["ade"].max() < 1e-2
    t, T = sequence(8.0, 0.0)
    w = windows(t, T, valid_refs(len(t)))
    pred = constant_velocity(np.tile([8.0, 0.0], (len(w["fut"]), 1)))
    assert displacement_metrics(pred, w["fut"])["fde"].max() < 1e-6


def test_metrics_lateral_vs_longitudinal():
    gt = np.zeros((1, FUT, 3))
    gt[0, :, 0] = np.arange(1, FUT + 1)
    pred = gt.copy()
    pred[0, :, 1] += 1.0  # 1 m to the left everywhere
    m = displacement_metrics(pred, gt)
    assert m["ade"][0] == pytest.approx(1.0) and m["lat"][0] == pytest.approx(1.0)
    assert m["lon"][0] == pytest.approx(0.0) and not m["miss"][0]


@pytest.mark.parametrize("speed,yaw_rate,accel,expected", [
    (10.0, 0.0, 0.0, "cruise"),
    (8.0, 0.25, 0.0, "turn_left"),     # 57 deg in 4 s
    (8.0, -0.25, 0.0, "turn_right"),
    (12.0, 0.06, 0.0, "curve"),         # 13.8 deg
    (12.0, 0.0, -2.0, "brake"),
    (5.0, 0.0, 2.0, "accel"),
    (0.0, 0.0, 0.0, "static"),
    (0.0, 0.0, 1.5, "start"),
])
def test_scenario_tags(speed, yaw_rate, accel, expected):
    t, T = sequence(speed, yaw_rate, n=HIST + FUT, accel=accel)
    w = windows(t, T, np.array([HIST - 1]))
    tags = tag_scenarios(w["fut"], np.array([speed]))
    assert tags["primary"][0] == expected


def test_lateral_shift_tag():
    # straight drive with a 3.5 m sideways S-shift over the 4 s future
    n = HIST + FUT
    t = np.arange(n) * 0.5
    x = 12.0 * t
    s = np.clip((t - t[HIST - 1]) / 4.0, 0, 1)
    y = 3.5 * (3 * s**2 - 2 * s**3)
    psi = np.arctan2(np.gradient(y), np.gradient(x))
    psi[-1] = 0.0
    T = np.stack([make_T(rot_z(p), [a, b, 0]) for a, b, p in zip(x, y, psi)])
    w = windows(t, T, np.array([HIST - 1]))
    assert tag_scenarios(w["fut"], np.array([12.0]))["primary"][0] == "lateral_shift"
