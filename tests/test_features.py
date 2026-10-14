import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from egopred import features, paths  # noqa: E402
from egopred.manifest import Manifest  # noqa: E402
from egopred.openscene import frames  # noqa: E402

from .test_frames import archive, jpeg  # noqa: E402,F401 - fixture reuse

TINY = features.FeatureConfig(name="tiny", model="vit_small_patch14_reg4_dinov2", input_hw=(28, 56),
                              grid=(2, 2), pca_dim=8, pca_frames=4)


def test_decode_resizes_to_backbone_input():
    out = features.decode(jpeg(w=192, h=108), (28, 56))
    assert out.shape == (3, 28, 56) and out.dtype == np.uint8


def test_pca_subset_is_deterministic_and_order_free():
    p = [f"log/CAM_F0/{i:016x}.jpg" for i in range(100)]
    a = features.pca_subset(p, 10)
    assert a == features.pca_subset(list(reversed(p)) + p[:5], 10) and len(a) == 10


def test_fit_pca_recovers_low_rank_data():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(16, 3)))[0].T  # 3 orthonormal directions in 16-d
    x = torch.from_numpy(rng.normal(size=(500, 3)) * [5, 3, 1] @ basis + 7.0)
    pca = features.fit_pca(x, 3)
    assert pca["explained"] > 0.999
    c = pca["components"]
    np.testing.assert_allclose(c @ c.T, np.eye(3), atol=1e-5)
    np.testing.assert_allclose(basis @ c.T @ c, basis, atol=1e-6)  # spans the data subspace
    np.testing.assert_allclose(pca["mean"], 7.0, atol=0.5)


def test_run_extracts_verifies_and_skips(archive):  # noqa: F811
    m = Manifest(paths.MANIFEST)
    frames.ingest_archive("mini", "a0", str(archive), m)
    idx = pd.read_parquet(frames.shard_paths("mini", "a0")[1])
    hist = {f"hist_f0_{k}": idx["f0_path"].to_numpy()[[k, k + 1, k + 2]] for k in range(4)}
    paths.SAMPLES.mkdir(parents=True)
    pd.DataFrame({**hist, "subset": "train"}).to_parquet(paths.SAMPLES / "t.parquet")

    stats = features.run(TINY, m, "cpu", batch=4, workers=0, pretrained=False, pca_train_table="t",
                         splits=("mini",))
    assert [s["frames"] for s in stats] == [len(idx)]
    cls_p, grid_p, index_p = features.feature_paths(TINY, "mini", "a0")
    cls, grid = np.load(cls_p), np.load(grid_p)
    assert cls.shape == (len(idx), 384) and grid.shape == (len(idx), 2, 2, 8)
    assert cls.dtype == grid.dtype == np.float16 and np.isfinite(grid).all()
    assert (pd.read_parquet(index_p)["f0_path"] == idx["f0_path"]).all()
    assert m.state("features:tiny", "mini/a0") == "verified"
    assert m.state("features_raw:tiny", "mini/a0") == "verified"
    raw_cls = np.load(features.raw_paths(TINY, "mini", "a0")[0])
    np.testing.assert_array_equal(raw_cls, cls)  # CLS kept at full width
    assert features.coverage(TINY, m, ["t"], splits=("mini",))["t"] == {"frames_needed": 6, "missing_frames": 0,
                                                       "missing_features": 0}
    store = features.FeatureStore(TINY, m)
    rev = idx["f0_path"].to_numpy()[::-1]
    np.testing.assert_array_equal(store.get(rev, "grid"), grid[::-1])
    assert idx["f0_path"][0] in store and "nope" not in store
    again = features.run(TINY, m, "cpu", batch=4, workers=0, pretrained=False, pca_train_table="t",
                         splits=("mini",))
    assert again == []  # verified shard skipped


def test_pca_waits_for_its_frames_but_backbone_runs(archive):  # noqa: F811
    m = Manifest(paths.MANIFEST)
    frames.ingest_archive("mini", "a0", str(archive), m)
    paths.SAMPLES.mkdir(parents=True)
    extra = [f"other/CAM_F0/{i:016x}.jpg" for i in range(8)]  # not ingested yet
    pd.DataFrame({**{f"hist_f0_{k}": extra for k in range(4)}, "subset": "train"}).to_parquet(
        paths.SAMPLES / "t.parquet")
    out = features.run(TINY, m, "cpu", batch=4, workers=0, pretrained=False, pca_train_table="t",
                       splits=("mini",))
    assert out == []
    assert m.state("features_raw:tiny", "mini/a0") == "verified"  # stage 1 done anyway
    assert m.state("features:tiny", "mini/a0") is None
    assert not (features.config_dir(TINY) / "pca.npz").exists()


def test_shards_still_arriving_are_skipped_then_processed(archive):  # noqa: F811
    m = Manifest(paths.MANIFEST)
    frames.ingest_archive("mini", "a0", str(archive), m)
    frames.ingest_archive("mini", "a1", str(archive), m)
    idx = pd.read_parquet(frames.shard_paths("mini", "a0")[1])
    hist = {f"hist_f0_{k}": idx["f0_path"].to_numpy()[[k, k + 1, k + 2]] for k in range(4)}
    paths.SAMPLES.mkdir(parents=True)
    pd.DataFrame({**hist, "subset": "train"}).to_parquet(paths.SAMPLES / "t.parquet")
    # a1 is listed as verified (manifest copied from the local server) but only half arrived
    tar1 = frames.shard_paths("mini", "a1")[0]
    full = tar1.read_bytes()
    tar1.write_bytes(full[: len(full) // 2])
    assert features.shard_ready("mini", "a0") and not features.shard_ready("mini", "a1")
    assert features.frame_shards(m, ("mini",)) == [("mini", "a0")]

    def run():
        return features.run(TINY, m, "cpu", batch=4, workers=0, pretrained=False, pca_train_table="t",
                            splits=("mini",))

    assert [s["shard"] for s in run()] == ["mini/a0"]
    assert features.pending_shards(TINY, m, ("mini",)) == ["mini/a1"]
    tar1.write_bytes(full)  # transfer completed
    assert [s["shard"] for s in run()] == ["mini/a1"]
    assert features.pending_shards(TINY, m, ("mini",)) == []
