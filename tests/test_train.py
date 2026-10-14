import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from egopred import paths  # noqa: E402
from egopred.model import ModelConfig, TrajectoryModel, fit_anchors, wta_loss  # noqa: E402
from egopred.openscene.samples import future, load_samples  # noqa: E402
from egopred.train import (Augment, Frames, Normalizer, Runner, TrainConfig, compare, curate,  # noqa: E402
                           export_predictions, future_frames, g3, hist_paths, kin_columns, multimodal_metrics,
                           run)
from egopred.trajectory import FUT, HIST, OFFSETS, constant_turn_rate  # noqa: E402

from .test_frames import archive, jpeg  # noqa: E402,F401 - fixture reuse

TINY = dict(d=32, layers=1, heads=4, k=3)


def synthetic_samples(n: int, seed: int = 0) -> pd.DataFrame:
    """Constant speed / yaw-rate drives: history and future consistent with each other."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(0, 15, n)
    w = rng.uniform(-0.3, 0.3, n)
    t = OFFSETS * 0.5  # -1.5 .. 4.0 s
    yaw = w[:, None] * t
    small = np.abs(w) < 1e-4
    ws = np.where(small, 1.0, w)[:, None]
    x = np.where(small[:, None], v[:, None] * t, v[:, None] * np.sin(yaw) / ws)
    y = np.where(small[:, None], 0.0, v[:, None] * (1 - np.cos(yaw)) / ws)
    d = {"sample_id": [f"s{i}" for i in range(n)], "subset": np.where(np.arange(n) % 5 == 0, "val", "train"),
         "gap": False, "pose_jump": False, "scenario": np.where(np.abs(w) > 0.15, "curve", "cruise"),
         "vx": v, "vy": 0.0, "ax": 0.0, "ay": 0.0, "speed": v, "yaw_rate": w,
         "command": rng.integers(-1, 4, n)}
    for k in range(HIST):
        d[f"hist_f0_{k}"] = [f"log{i}/f{i + k}.jpg" for i in range(n)]
        d[f"hist_x_{k}"], d[f"hist_y_{k}"], d[f"hist_yaw_{k}"] = x[:, k], y[:, k], yaw[:, k]
    for k in range(FUT):
        d[f"fut_x_{k}"], d[f"fut_y_{k}"], d[f"fut_yaw_{k}"] = x[:, HIST + k], y[:, HIST + k], yaw[:, HIST + k]
    return pd.DataFrame(d)


@pytest.fixture
def tables():
    paths.SAMPLES.mkdir(parents=True)
    synthetic_samples(200).to_parquet(paths.SAMPLES / "navtrain.parquet")
    synthetic_samples(40, seed=1).drop(columns="subset").to_parquet(paths.SAMPLES / "navtest.parquet")


def test_synthetic_future_matches_ctrv():
    s = synthetic_samples(10)
    fut = np.stack([s[[f"fut_{c}_{k}" for c in "x y yaw".split()]].to_numpy() for k in range(FUT)], axis=1)
    np.testing.assert_allclose(constant_turn_rate(s["speed"].to_numpy(), s["yaw_rate"].to_numpy()), fut, atol=1e-6)


def test_multimodal_metrics_min_over_modes_and_top1():
    gt = np.zeros((2, FUT, 3))
    gt[:, :, 0] = np.arange(1, FUT + 1)
    traj = np.repeat(gt[:, None], 2, axis=1).copy()
    traj[:, 1, :, 1] += 3.0  # mode 1 is 3 m off laterally
    probs = np.array([[0.2, 0.8], [0.9, 0.1]])
    m = multimodal_metrics(traj, probs, gt)
    np.testing.assert_allclose(m["min_ade"], 0.0)
    np.testing.assert_allclose(m["ade"], [3.0, 0.0])  # top-1 = mode 1 for sample 0
    assert m["miss"].tolist() == [True, False] and m["top_is_best"].tolist() == [False, True]
    np.testing.assert_allclose(m["spread"], 3.0 / 2)  # mean over the 2x2 pairs, diagonal = 0


def test_wta_loss_trains_only_the_closest_mode():
    cfg = ModelConfig(k=2)
    target = torch.zeros(1, FUT, 3)
    traj = torch.zeros(1, 2, FUT, 3, requires_grad=True)
    with torch.no_grad():
        traj[0, 1] += 1.0
    logits = torch.zeros(1, 2, requires_grad=True)
    loss, parts = wta_loss(traj, logits, target, cfg)
    loss.backward()
    assert traj.grad[0, 1].abs().sum() == 0  # the far mode gets no regression gradient
    assert logits.grad[0, 0] < 0 < logits.grad[0, 1]  # CE pushes towards the closest mode


def test_wta_with_anchors_targets_the_nearest_anchor():
    cfg = ModelConfig(k=2, anchors=True)
    target = torch.zeros(1, FUT, 3)
    anchors = torch.zeros(2, FUT, 3)
    anchors[0, :, 1] = 5.0  # anchor 0 is 5 m off laterally, anchor 1 is on the target
    traj = torch.zeros(1, 2, FUT, 3, requires_grad=True)
    with torch.no_grad():
        traj[0, 0] += 0.0  # mode 0 happens to predict the target exactly...
        traj[0, 1, :, 0] += 1.0  # ...mode 1 is 1 m off: plain WTA would pick mode 0
    logits = torch.zeros(1, 2, requires_grad=True)
    loss, _ = wta_loss(traj, logits, target, cfg, anchors)
    loss.backward()
    assert traj.grad[0, 0].abs().sum() == 0 and traj.grad[0, 1].abs().sum() > 0
    assert logits.grad[0, 1] < 0 < logits.grad[0, 0]


def test_fit_anchors_finds_the_clusters():
    rng = np.random.default_rng(0)
    t = np.arange(1, FUT + 1)[None, :] * 0.5
    straight = np.stack([t * 10, 0 * t, 0 * t], -1).repeat(50, 0)
    left = np.stack([t * 8, t ** 2, 0.1 * t], -1).repeat(50, 0)
    fut = np.concatenate([straight, left]) + rng.normal(0, 0.05, (100, FUT, 3))
    a = fit_anchors(fut, 2)
    assert a.shape == (2, FUT, 3)
    end_y = sorted(a[:, -1, 1])
    np.testing.assert_allclose(end_y, [0.0, (FUT * 0.5) ** 2], atol=0.1)


def test_model_adds_anchors_to_its_offsets():
    cfg = ModelConfig(variant="A", anchors=True, **TINY)
    kin = len(kin_columns("A")) + 4
    m = TrajectoryModel(cfg, kin).eval()  # no dropout: the two calls must match
    a = np.random.default_rng(0).normal(size=(cfg.k, FUT, 3)).astype(np.float32)
    traj0, _ = m(torch.zeros(2, kin))
    m.set_anchors(a)
    traj1, _ = m(torch.zeros(2, kin))
    torch.testing.assert_close(traj1 - traj0, torch.from_numpy(a)[None].expand(2, -1, -1, -1))


def test_g3_overfits_with_anchors(tables):
    res = g3(n=64, epochs=150, device="cpu", model=ModelConfig(variant="C", dropout=0.0, anchors=True, **TINY),
             target_ade=0.5, batch=8)
    assert res["passed"], res


@pytest.mark.parametrize("variant,vision", [("A", "grid"), ("B", "cls"), ("C", "grid")])
def test_model_shapes(variant, vision):
    cfg = ModelConfig(variant=variant, vision=vision, **TINY)
    kin = len(kin_columns(variant)) + 4
    m = TrajectoryModel(cfg, kin, cls_dim=8, grid_dim=4, n_grid=6)
    traj, logits = m(torch.zeros(5, kin), torch.zeros(5, HIST, 8), torch.zeros(5, HIST, 6, 4))
    assert traj.shape == (5, 3, FUT, 3) and logits.shape == (5, 3)


def test_g3_overfits_with_fake_features(tables):
    res = g3(n=64, epochs=150, device="cpu", model=ModelConfig(variant="C", dropout=0.0, **TINY),
             target_ade=0.5, batch=8)
    assert res["passed"], res
    assert res["last_loss"] < res["first_loss"] / 5


def test_run_all_variants_and_compare(tables):
    names = []
    for v in "ABC":
        cfg = TrainConfig(name=f"t{v}", model=ModelConfig(variant=v, **TINY), epochs=2, batch=32)
        res = run(cfg, lambda ts: Frames.fake(hist_paths(*ts), 8, (2, 3, 4)), "cpu")
        assert {"val", "test"} <= set(res) and res["train_samples"] == 160
        assert (paths.OUT / "runs" / f"t{v}" / "best.pt").exists()
        names.append(f"t{v}")
    table = compare(names, "test")
    assert "B0-CTRV ADE" in table and "tC minADE" in table and "| curve |" in table
    # B0-CTRV is exact on constant-turn-rate data
    r = json.loads((paths.OUT / "runs" / "tC" / "result.json").read_text())
    assert pd.DataFrame(r["test"]["by_scenario"]).set_index("scenario").loc["ALL", "ctrv_ade"] < 1e-6
    # the exported predictions reproduce the evaluation of the best checkpoint
    fake = lambda ts: Frames.fake(hist_paths(*ts), 8, (2, 3, 4))  # noqa: E731
    counts = export_predictions("tC", fake, "cpu")
    p = np.load(paths.OUT / "runs" / "tC" / "predictions_test.npz")
    assert p["traj"].shape == (counts["test"], 3, FUT, 3) and p["probs"].shape == (counts["test"], 3)
    per = pd.read_parquet(paths.OUT / "runs" / "tC" / "per_sample_test.parquet")
    assert list(p["sample_id"]) == per["sample_id"].tolist()
    gt = future(load_samples("navtest").set_index("sample_id").loc[p["sample_id"]])
    top = p["probs"].astype(np.float32).argmax(1)
    ade = np.linalg.norm(p["traj"][np.arange(len(top)), top, :, :2].astype(np.float32) - gt[:, :, :2], axis=-1).mean(1)
    np.testing.assert_allclose(ade, per["ade"], atol=0.05)


def test_run_on_stored_features(archive):  # noqa: F811
    """The real path: frame shard -> stage 1+2 features on disk -> FeatureStore -> training."""
    from egopred import features
    from egopred.manifest import Manifest
    from egopred.openscene import frames

    from .test_features import TINY as FEAT

    m = Manifest(paths.MANIFEST)
    frames.ingest_archive("mini", "a0", str(archive), m)
    f0 = pd.read_parquet(frames.shard_paths("mini", "a0")[1])["f0_path"].to_numpy()
    paths.SAMPLES.mkdir(parents=True)
    for name, n, seed in (("navtrain", 60, 0), ("navtest", 12, 1)):
        s = synthetic_samples(n, seed)
        for k in range(HIST):
            s[f"hist_f0_{k}"] = f0[(np.arange(n) + k) % len(f0)]
        s.drop(columns="subset" if name == "navtest" else []).to_parquet(paths.SAMPLES / f"{name}.parquet")
    features.run(FEAT, m, "cpu", batch=4, workers=0, pretrained=False, splits=("mini",))
    store = features.FeatureStore(FEAT, m)
    cfg = TrainConfig(name="real", model=ModelConfig(variant="C", **TINY), epochs=2, batch=16)
    res = run(cfg, lambda ts: Frames.from_store(store, hist_paths(*ts)), "cpu")
    overall = pd.DataFrame(res["test"]["by_scenario"]).set_index("scenario").loc["ALL"]
    assert res["train_samples"] == 48 and np.isfinite(overall["C_ade"]) and np.isfinite(overall["C_min_ade"])


def test_same_seed_same_weights():
    cfg = ModelConfig(variant="A", **TINY)
    kin = synthetic_samples(4)
    norm = Normalizer.fit(kin[kin_columns("A")].to_numpy(np.float64))
    a, b = (Runner(cfg, Frames.empty(), norm, "cpu", seed=3) for _ in range(2))
    for pa, pb in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(pa, pb)


def test_curate_drops_only_trivial_samples():
    n = 400
    t = np.arange(1, FUT + 1) * 0.5
    df = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "speed": 10.0, "yaw_rate": 0.0,
                       "vx": 10.0, "vy": 0.0})
    for k in range(FUT):  # first half: straight at 10 m/s (CTRV exact), second half: stops
        df[f"fut_x_{k}"] = np.where(np.arange(n) < n // 2, 10 * t[k], 0.0)
        df[f"fut_y_{k}"] = 0.0
        df[f"fut_yaw_{k}"] = 0.0
    kept = curate(df, keep=0.25, max_ade=0.5)
    idx = kept["sample_id"].str[1:].astype(int)
    assert (idx >= n // 2).sum() == n // 2  # every non-trivial sample stays
    assert 25 < (idx < n // 2).sum() < 75  # ~25% of the 200 trivial ones
    assert curate(df, keep=1.0, max_ade=0.5) is df


def _vision_runner():
    s = synthetic_samples(40)
    frames = Frames.fake(hist_paths(s), 8, (2, 3, 4))
    norm = Normalizer.fit(s[kin_columns("C")].to_numpy(np.float64))
    r = Runner(ModelConfig(variant="C", **TINY), frames, norm, "cpu")
    rows = torch.from_numpy(frames.rows(s))
    return r, r.cls[rows].float(), r.grid[rows].float()


def test_augment_masks_and_keeps_the_current_frame():
    r, cls, grid = _vision_runner()
    c, g, kf, kc = r._augment(cls, grid, Augment(frame_drop=1.0))
    assert torch.equal(c, cls) and torch.equal(g, grid) and kc is None
    assert not kf[:, :-1].any() and kf[:, -1].all()  # the current frame is never dropped
    _, _, kf, _ = r._augment(cls, grid, Augment(vision_drop=1.0))
    assert not kf.any()
    _, _, kf, kc = r._augment(cls, grid, Augment(token_drop=0.5))
    assert kf.all() and kc.shape == grid.shape[:-1] and 0.3 < (~kc).float().mean() < 0.7
    c, _, _, _ = r._augment(cls, grid, Augment(noise=0.1))
    assert 0 < (c - cls).abs().mean() < 0.5 * r.stats["cls"][1].mean()


def test_masked_tokens_do_not_affect_the_output():
    cfg = ModelConfig(variant="C", **TINY)
    kin = len(kin_columns("C")) + 4
    m = TrajectoryModel(cfg, kin, cls_dim=8, grid_dim=4, n_grid=6).eval()
    x, c, g = torch.randn(5, kin), torch.randn(5, HIST, 8), torch.randn(5, HIST, 6, 4)
    kf = torch.ones(5, HIST, dtype=torch.bool)
    kf[:, 0] = False
    kc = torch.rand(5, HIST, 6) > 0.5
    out = m(x, c, g, kf, kc)
    c2, g2 = c.clone(), g.clone()
    c2[:, 0] += 10.0  # dropped frame
    g2[~kc] += 10.0  # dropped cells
    out2 = m(x, c2, g2, kf, kc)
    torch.testing.assert_close(out[0], out2[0]) and torch.testing.assert_close(out[1], out2[1])
    assert not torch.allclose(m(x, c2, g2)[0], out[0])  # without masks they matter


def test_run_with_augmentation(tables):
    cfg = TrainConfig(name="taug", model=ModelConfig(variant="C", **TINY), epochs=2, batch=32,
                      aug=Augment(noise=0.2, frame_drop=0.3, token_drop=0.2, vision_drop=0.1))
    res = run(cfg, lambda ts: Frames.fake(hist_paths(*ts), 8, (2, 3, 4)), "cpu")
    assert res["config"]["aug"]["frame_drop"] == 0.3 and np.isfinite(res["history"][-1]["loss"])


def one_log(n: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic samples as consecutive 2 Hz windows of one log: sample i's current frame is f(i+3)."""
    s = synthetic_samples(n, seed).assign(log_name="log0", timestamp=np.arange(n) * 500_000)
    for k in range(HIST):
        s[f"hist_f0_{k}"] = [f"log0/f{i + k}.jpg" for i in range(n)]
    return s


def test_future_frames_finds_the_frame_dt_later():
    s = one_log(10)
    out = future_frames(s, s, (1.0, 0.25))
    assert out["fut_f0_0"][:8].tolist() == [f"log0/f{i + 5}.jpg" for i in range(8)]
    assert out["fut_f0_0"][8] is None and out["fut_f0_0"][9] is None  # beyond the last frame
    assert all(p is None for p in out["fut_f0_1"])  # no frame within 0.15 s of t0 + 0.25 s
    other = s.assign(log_name="log1")
    assert all(p is None for p in future_frames(other, s, (1.0,))["fut_f0_0"])  # never from another log


@pytest.mark.parametrize("vision", ["grid", "cls"])
def test_future_head_shapes_and_conditioning(vision):
    cfg = ModelConfig(variant="C", vision=vision, future=(1.0, 2.0), future_cond=True, **TINY)
    kin = len(kin_columns("C")) + 4
    m = TrajectoryModel(cfg, kin, cls_dim=8, grid_dim=4, n_grid=6).eval()
    x, c, g = torch.randn(5, kin), torch.randn(5, HIST, 8), torch.randn(5, HIST, 6, 4)
    traj, _, memory, mask = m(x, c, g, return_memory=True)
    pc, pg = m.predict_future(memory, mask, traj[:, 0])
    assert pc.shape == (5, 2, 8) and (pg.shape == (5, 2, 6, 4) if vision == "grid" else pg is None)
    pc2, _ = m.predict_future(memory, mask, traj[:, 0] + 1.0)
    assert not torch.allclose(pc, pc2)  # conditioned on the trajectory


def test_run_with_future_loss():
    paths.SAMPLES.mkdir(parents=True)
    one_log(200).to_parquet(paths.SAMPLES / "navtrain.parquet")
    one_log(40, seed=1).drop(columns="subset").to_parquet(paths.SAMPLES / "navtest.parquet")
    frames_for = lambda ts: Frames.fake(hist_paths(*ts), 8, (2, 3, 4))  # noqa: E731
    cfg = TrainConfig(name="tfut", model=ModelConfig(variant="C", future=(1.0, 2.0), future_weight=1.0, **TINY),
                      epochs=3, batch=32, aug=Augment(noise=0.2, frame_drop=0.3))
    res = run(cfg, frames_for, "cpu")
    h = res["history"]
    assert all(np.isfinite(r["fut"]) for r in h) and h[-1]["fut"] < h[0]["fut"] and h[-1]["fut_r2"] > h[0]["fut_r2"]
    assert export_predictions("tfut", frames_for, "cpu") == {"val": 40, "test": 40}  # checkpoint with the head loads
