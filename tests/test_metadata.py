import io
import pickle
import tarfile

import numpy as np
import pytest

from egopred.geometry import make_T
from egopred.manifest import Manifest
from egopred.openscene import metadata
from egopred.openscene.metadata import array


def fake_log(log_name, n=20, speed=5.0, t0=1_600_000_000_000_000):
    recs = []
    for i in range(n):
        x = speed * 0.5 * i
        recs.append({
            "token": f"{log_name[-4:]}{i:012x}", "frame_idx": i, "timestamp": t0 + i * 500_000,
            "log_name": log_name, "log_token": "lt", "scene_name": "s", "scene_token": "st",
            "map_location": "us-nv-las-vegas-strip", "vehicle_name": "veh-35",
            "sample_prev": None, "sample_next": None,
            "ego2global": make_T(np.eye(3), [600000.0 + x, 4e6, 600.0]),
            "lidar2ego": np.eye(4), "ego_dynamic_state": [speed, 0.0, 0.0, 0.0],
            "can_bus": np.zeros(18), "driving_command": np.array([0, 1, 0, 0]),
            "cams": {"CAM_F0": {"data_path": f"{log_name}/CAM_F0/{i:016x}.jpg",
                                "cam_intrinsic": np.eye(3), "distortion": np.zeros(5),
                                "sensor2lidar_rotation": np.eye(3), "sensor2lidar_translation": np.zeros(3)},
                     "CAM_L0": {"data_path": "ignored"}},
            "anns": {"gt_names": np.array(["vehicle", "pedestrian"])},
        })
    # shuffled on purpose: compact_log must sort by time
    return recs[::-1]


def test_compact_log_fields_and_order():
    df = metadata.compact_log(fake_log("2021.05.12.22.00.38_veh-35_00001_00100"))
    assert df["timestamp"].is_monotonic_increasing and len(df) == 20
    assert array(df, "e2g", (4, 4))[3, 0, 3] == pytest.approx(600000.0 + 7.5)
    assert df["cmd_1"].tolist() == [1] * 20 and df["n_boxes"].iloc[0] == 2
    assert df["f0_path"].iloc[0].endswith("CAM_F0/0000000000000000.jpg")
    assert metadata.check_log(df) == []


def test_ingest_metadata_from_stream(tmp_path):
    src = tmp_path / "meta.tgz"
    logs = ["2021.05.12.22.00.38_veh-35_00001_00100", "2021.06.09.17.23.18_veh-38_00002_00200"]
    with tarfile.open(src, "w:gz") as tf:
        for name in logs:
            data = pickle.dumps(fake_log(name))
            info = tarfile.TarInfo(f"openscene-v1.1/meta_datas/mini/{name}.pkl")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    m = Manifest(metadata.paths.MANIFEST)
    metadata.ingest_metadata("mini", m, src=str(src))
    assert m.get("meta_archive", "mini") == ("verified", {"src": str(src), "logs": 2,
                                                         "src_bytes": src.stat().st_size, "frames": 40})
    assert [f.stem for f in metadata.log_files("mini")] == logs
    # rerun skips without touching the source
    metadata.ingest_metadata("mini", m, src="/nonexistent")
