import io
import tarfile

import numpy as np
import pytest
from PIL import Image

from egopred.manifest import Manifest
from egopred.openscene import frames


def jpeg(w=64, h=36, seed=0):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def make_tgz(path, members):
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


@pytest.fixture
def archive(tmp_path):
    members = []
    for log in ["2021.05.12.22.00.38_veh-35_01008_01518", "2021.06.09.17.23.18_veh-38_02526_03027"]:
        for cam in ["CAM_F0", "CAM_L0", "CAM_B0"]:
            for i in range(3):
                members.append((f"openscene-v1.1/sensor_blobs/mini/{log}/{cam}/{i:016x}.jpg", jpeg(seed=i)))
        members.append((f"openscene-v1.1/sensor_blobs/mini/{log}/MergedPointCloud/0.pcd", b"pcd"))
    p = tmp_path / "src.tgz"
    make_tgz(p, members)
    return p


def test_front_relpath():
    assert frames.front_relpath("navtrain_current_1/2021.05.12_veh-35_1_2/CAM_F0/ab12.jpg") == \
        "2021.05.12_veh-35_1_2/CAM_F0/ab12.jpg"
    assert frames.front_relpath("x/log/CAM_F0R/ab.jpg") is None
    assert frames.front_relpath("x/log/CAM_L0/ab.jpg") is None


def test_ingest_keeps_only_front_and_verifies(archive):
    m = Manifest(frames.paths.MANIFEST)
    frames.ingest_archive("mini", "a0", str(archive), m)
    state, info = m.get("frames", "mini/a0")
    assert state == "verified" and info["frames"] == 6 and info["logs"] == 2
    tar_path, index_path = frames.shard_paths("mini", "a0")
    import pandas as pd

    idx = pd.read_parquet(index_path)
    assert all("/CAM_F0/" in p for p in idx["f0_path"])
    row = idx.iloc[4]
    Image.open(io.BytesIO(frames.read_frame(tar_path, row["offset"], row["size"]))).verify()
    # idempotent: a verified shard is skipped
    frames.ingest_archive("mini", "a0", "/nonexistent", m)


def test_wanted_filter_and_resize(archive):
    m = Manifest(frames.paths.MANIFEST)
    wanted = {"2021.05.12.22.00.38_veh-35_01008_01518/CAM_F0/0000000000000001.jpg"}
    frames.ingest_archive("mini", "a1", str(archive), m, wanted=wanted, max_width=32)
    tar_path, index_path = frames.shard_paths("mini", "a1")
    import pandas as pd

    idx = pd.read_parquet(index_path)
    assert idx["f0_path"].tolist() == sorted(wanted)
    im = Image.open(io.BytesIO(frames.read_frame(tar_path, idx["offset"][0], idx["size"][0])))
    assert im.size == (32, 18)


def test_corrupt_frame_fails_verification(tmp_path):
    p = tmp_path / "bad.tgz"
    make_tgz(p, [("x/log_a/CAM_F0/aa.jpg", jpeg()), ("x/log_a/CAM_F0/bb.jpg", b"not a jpeg")])
    m = Manifest(frames.paths.MANIFEST)
    with pytest.raises(RuntimeError, match="verification"):
        frames.ingest_archive("mini", "bad", str(p), m)
    assert m.state("frames", "mini/bad") == "failed"


def test_stream_cut_is_retried(archive, tmp_path, monkeypatch):
    truncated = tmp_path / "cut.tgz"
    truncated.write_bytes(archive.read_bytes()[: archive.stat().st_size // 2])
    calls = []

    def flaky_open(src):
        calls.append(src)
        return frames.open_source.__wrapped__(str(truncated) if len(calls) == 1 else src)

    flaky_open.__wrapped__ = frames.open_source
    monkeypatch.setattr(frames, "open_source", flaky_open)
    monkeypatch.setattr(frames, "RETRY_WAIT_S", 0.0)
    m = Manifest(tmp_path / "m.sqlite")
    frames.ingest_archive("mini", "a0", str(archive), m)
    assert len(calls) == 2 and m.state("frames", "mini/a0") == "verified"


def test_persistent_stream_error_marks_failed(archive, tmp_path, monkeypatch):
    truncated = tmp_path / "cut.tgz"
    truncated.write_bytes(archive.read_bytes()[: archive.stat().st_size // 2])
    monkeypatch.setattr(frames, "RETRY_WAIT_S", 0.0)
    m = Manifest(tmp_path / "m.sqlite")
    with pytest.raises(frames.STREAM_ERRORS):
        frames.ingest_archive("mini", "a0", str(truncated), m)
    assert m.state("frames", "mini/a0") == "failed"


def test_adopt_verifies_copied_shards(archive, tmp_path):
    from egopred import paths

    src = Manifest(tmp_path / "remote.sqlite")
    frames.ingest_archive("mini", "a0", str(archive), src)
    frames.ingest_archive("mini", "a1", str(archive), src)
    tar1 = frames.shard_paths("mini", "a1")[0]
    data = bytearray(tar1.read_bytes())
    data[len(data) // 2:] = b"\0" * (len(data) - len(data) // 2)  # damaged copy
    tar1.write_bytes(bytes(data))
    local = Manifest(paths.MANIFEST)
    assert frames.adopt_shards("mini", local) == ["mini/a0"]
    assert local.state("frames", "mini/a0") == "verified" and local.state("frames", "mini/a1") == "failed"
