import numpy as np

import pandas as pd

from egopred.figures import pick_window
from egopred.geometry import inv_T, project, transform_points
from egopred.viz import perspective_view, virtual_camera


def test_virtual_camera_looks_forward_and_down():
    T, K = virtual_camera()
    pts = np.array([[10.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    uv, ok = project(transform_points(inv_T(T), pts), K)
    assert ok.all()
    assert abs(uv[0, 0] - K[0, 2]) < 1e-6  # straight ahead stays on the image column centre
    assert uv[1, 1] < uv[0, 1]  # farther points are higher in the image


def test_perspective_view_size():
    path = np.stack([np.arange(1, 9) * 4.0, np.zeros(8), np.zeros(8)], 1)
    im = perspective_view([(path, (255, 0, 0), "a")], size=(200, 120), ss=1)
    assert im.size == (200, 120)


def test_pick_window_gain_loss_random():
    n = 12  # one log, 2 Hz, all moving; C beats A on samples 2-4, A beats C on samples 7-9
    smp = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "log_name": "log0",
                        "timestamp": np.arange(n) * 500_000, "speed": 5.0, "gap": False, "pose_jump": False,
                        "scenario": "cruise"})
    ade_a = np.ones(n)
    ade_a[2:5], ade_a[7:10] = 3.0, 0.2
    a = pd.DataFrame({"ade": ade_a}, index=smp["sample_id"])
    c = pd.DataFrame({"ade": np.ones(n)}, index=smp["sample_id"])
    assert pick_window(a, c, smp, 3, "gain")["sample_id"].tolist() == ["s2", "s3", "s4"]
    assert pick_window(a, c, smp, 3, "loss")["sample_id"].tolist() == ["s7", "s8", "s9"]
    w = pick_window(a, c, smp, 3, "random", seed=1)
    assert len(w) == 3 and (np.diff(w["timestamp"]) == 500_000).all()
