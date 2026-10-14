"""Training and evaluation of the A/B/C variants on precomputed features (never on video).

All per-frame features of a run are gathered once into two device-resident arrays (CLS and
grid, float16; ~5.5 GB for NAVSIM, fits a 24 GB GPU), and each sample holds the row indices
of its HIST frames, so batches are plain index gathers with no data loader in the way.

Evaluation reports, per scenario and next to the B0 baselines: top-1 (most probable mode)
ADE/FDE/miss/lateral/longitudinal, minADE/minFDE/miss over the K modes, the spread of the
modes (endpoint distance) and the calibration of the mode probabilities.
"""

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .g1 import evaluate as b0_metrics
from .g1 import md_table, stratified
from .model import VARIANTS, ModelConfig
from .openscene.samples import future, load_samples
from .trajectory import DT, FUT, HIST, displacement_metrics

log = logging.getLogger(__name__)

HIST_KIN = [f"hist_{c}_{k}" for k in range(HIST - 1) for c in ("x", "y", "yaw")]  # t0 is the origin
DYN_KIN = ["vx", "vy", "ax", "ay", "speed", "yaw_rate"]
N_COMMANDS = 4
TOP1 = ["ade", "fde", "miss", "lat", "lon"]


@dataclass
class Augment:
    """Train-time augmentation of the visual inputs, against the overfitting of the vision branch.
    Dropped tokens are masked out of the attention (filling them with the feature mean teaches
    the grid model to rely on the fill)."""
    noise: float = 0.0  # gaussian noise, in units of the per-dimension feature std
    frame_drop: float = 0.0  # per history frame except the current one
    token_drop: float = 0.0  # per grid cell
    vision_drop: float = 0.0  # all visual inputs of a sample (the model must fall back on kinematics)

    def active(self) -> bool:
        return any((self.noise, self.frame_drop, self.token_drop, self.vision_drop))


@dataclass
class TrainConfig:
    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    train_table: str = "navtrain"
    test_table: str = "navtest"
    fraction: float = 1.0  # of the train samples (deterministic, by sample id)
    epochs: int = 30
    batch: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_minutes: float = 0.0  # stop training after this many minutes (0: no limit)
    trivial_keep: float = 1.0  # share of the trivial train samples kept (curation ablation)
    trivial_ade: float = 0.5  # trivial = B0-CTRV already within this ADE (m)
    aug: Augment = field(default_factory=Augment)

    seed: int = 0


# ---------------------------------------------------------------- data

def kin_columns(variant: str) -> list[str]:
    kin, _ = VARIANTS[variant]
    return ["speed"] if kin == "speed" else HIST_KIN + DYN_KIN


def select(s: pd.DataFrame, subset: str | None, fraction: float = 1.0, drop_flagged: bool = False) -> pd.DataFrame:
    """Samples of a subset ("train"/"val", None = all). Train drops QC-flagged samples
    (timestamp gaps, pose jumps); evaluation keeps them, so numbers match the benchmark set."""
    if subset is not None:
        s = s[s["subset"] == subset]
    if drop_flagged:
        s = s[~(s["gap"] | s["pose_jump"])]
    if fraction < 1.0:
        s = s[unit_hash(s["sample_id"]) < fraction]
    return s.reset_index(drop=True)


def unit_hash(ids: pd.Series, salt: str = "") -> pd.Series:
    """Deterministic uniform [0, 1) value per sample id."""
    return ids.map(lambda t: int(hashlib.md5((salt + t).encode()).hexdigest()[:8], 16) / 2**32)


def curate(s: pd.DataFrame, keep: float, max_ade: float) -> pd.DataFrame:
    """Keep only `keep` of the trivial samples (B0-CTRV ADE below max_ade), chosen by sample id;
    the others are all kept."""
    if keep >= 1.0:
        return s
    trivial = b0_metrics(s)["ctrv_ade"].to_numpy() < max_ade
    kept = ~trivial | (unit_hash(s["sample_id"], "curate:").to_numpy() < keep)
    return s[kept].reset_index(drop=True)


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        return cls(x.mean(axis=0), x.std(axis=0) + 1e-6)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def kin_inputs(s: pd.DataFrame, cols: list[str], norm: Normalizer) -> np.ndarray:
    onehot = np.zeros((len(s), N_COMMANDS), np.float32)
    cmd = s["command"].to_numpy()
    onehot[cmd >= 0, cmd[cmd >= 0]] = 1.0  # -1: no command
    return np.concatenate([norm(s[cols].to_numpy(np.float64)).astype(np.float32), onehot], axis=1)


class Frames:
    """CLS and grid features of the frames used by a run, gathered once, row-indexed."""

    def __init__(self, f0_paths, cls: np.ndarray, grid: np.ndarray):
        self.row = pd.Series(np.arange(len(f0_paths)), index=pd.Index(f0_paths))
        self.cls, self.grid = cls, grid.reshape(len(grid), int(np.prod(grid.shape[1:-1])), grid.shape[-1])

    @classmethod
    def from_store(cls, store, f0_paths) -> "Frames":
        f0_paths = list(dict.fromkeys(f0_paths))
        return cls(f0_paths, store.get(f0_paths, "cls"), store.get(f0_paths, "grid"))

    @classmethod
    def fake(cls, f0_paths, cls_dim: int = 768, grid_shape=(6, 10, 256)) -> "Frames":
        """Deterministic random features per frame path: for gate G3 and tests, where the model
        must memorise samples, before (or without) real features."""
        f0_paths = list(dict.fromkeys(f0_paths))
        seeds = [int(hashlib.md5(p.encode()).hexdigest()[:8], 16) for p in f0_paths]
        c = np.stack([np.random.default_rng(x).standard_normal(cls_dim) for x in seeds]).astype(np.float16)
        g = np.stack([np.random.default_rng(x + 1).standard_normal(grid_shape) for x in seeds]).astype(np.float16)
        return cls(f0_paths, c, g)

    @classmethod
    def empty(cls) -> "Frames":
        """No vision input (variant A)."""
        return cls([], np.zeros((0, 1), np.float16), np.zeros((0, 1, 1), np.float16))

    def rows(self, s: pd.DataFrame) -> np.ndarray:
        return np.stack([self.row.loc[s[f"hist_f0_{k}"]].to_numpy() for k in range(HIST)], axis=1)

    def lookup(self, f0_paths) -> np.ndarray:
        """Row of each frame path, -1 where missing (None or not loaded)."""
        return self.row.reindex(pd.Index(f0_paths)).fillna(-1).to_numpy(np.int64)


def hist_paths(*tables: pd.DataFrame) -> list[str]:
    return [p for s in tables for k in range(HIST) for p in s[f"hist_f0_{k}"]]


def future_frames(s: pd.DataFrame, pool: pd.DataFrame, dts, tol: float = 0.15) -> dict[str, np.ndarray]:
    """Columns fut_f0_<i>: path of the frame of the same log nearest t0 + dts[i] seconds (within
    tol s) among the history frames of the samples in `pool`, None where there is none. Frame
    times are exact for current frames and t0 - k * DT for older history frames."""
    frames = pd.concat([pd.DataFrame({"log": pool["log_name"].to_numpy(), "f0": pool[f"hist_f0_{k}"].to_numpy(),
                                      "t": pool["timestamp"].to_numpy() - int((HIST - 1 - k) * DT * 1e6)})
                        for k in reversed(range(HIST))]).drop_duplicates("f0")  # current frames first: exact time
    by_log = {log: g.sort_values("t") for log, g in frames.groupby("log")}
    out = {f"fut_f0_{i}": np.full(len(s), None, dtype=object) for i in range(len(dts))}
    for log, idx in s.groupby("log_name").indices.items():
        if log not in by_log:
            continue
        g = by_log[log]
        ts, f0 = g["t"].to_numpy(), g["f0"].to_numpy()
        for i, dt in enumerate(dts):
            want = s["timestamp"].to_numpy()[idx] + int(dt * 1e6)
            j = np.clip(np.searchsorted(ts, want), 1, max(1, len(ts) - 1))
            j = np.where(np.abs(ts[j - 1] - want) < np.abs(ts[np.minimum(j, len(ts) - 1)] - want), j - 1,
                         np.minimum(j, len(ts) - 1))
            ok = np.abs(ts[j] - want) < tol * 1e6
            out[f"fut_f0_{i}"][idx[ok]] = f0[j[ok]]
    return out


# ---------------------------------------------------------------- model I/O

class Runner:
    """Model + device-resident inputs for one variant; gathers batches by index."""

    def __init__(self, mcfg: ModelConfig, frames: Frames, norm: Normalizer, device: str, seed: int = 0):
        import torch

        from .model import TrajectoryModel

        torch.manual_seed(seed)  # weight init: runs of the same config must be comparable

        self.torch, self.device, self.norm, self.mcfg = torch, device, norm, mcfg
        self.cols = kin_columns(mcfg.variant)
        _, vision = VARIANTS[mcfg.variant]
        self.cls = torch.from_numpy(frames.cls).to(device) if vision else None
        self.grid = torch.from_numpy(frames.grid).to(device) if vision and mcfg.vision == "grid" else None
        self.frames = frames
        self.stats = self._feature_stats() if self.cls is not None else None
        self.model = TrajectoryModel(mcfg, len(self.cols) + N_COMMANDS, frames.cls.shape[1],
                                     frames.grid.shape[-1], frames.grid.shape[1]).to(device)

    def tensors(self, s: pd.DataFrame) -> dict:
        t = self.torch
        out = {"kin": t.from_numpy(kin_inputs(s, self.cols, self.norm)).to(self.device),
               "target": t.from_numpy(future(s).astype(np.float32)).to(self.device)}
        if self.cls is not None:
            out["rows"] = t.from_numpy(self.frames.rows(s)).to(self.device)
        return out

    def forward(self, data: dict, idx, aug: Augment | None = None, memory: bool = False):
        rows = data["rows"][idx] if "rows" in data else None
        cls = self.cls[rows].float() if self.cls is not None else None
        grid = self.grid[rows].float() if self.grid is not None else None
        keep_frames = keep_cells = None
        if aug is not None and cls is not None and aug.active():
            cls, grid, keep_frames, keep_cells = self._augment(cls, grid, aug)
        return self.model(data["kin"][idx], cls, grid, keep_frames, keep_cells, memory)

    def future_rows(self, s: pd.DataFrame) -> "torch.Tensor":
        """(M, n_fut) rows of the future frames of the samples (columns fut_f0_<i>), -1 where missing."""
        rows = np.stack([self.frames.lookup(s[f"fut_f0_{i}"]) for i in range(len(self.mcfg.future))], axis=1)
        return self.torch.from_numpy(rows).to(self.device)

    def future_loss(self, memory, mask, data: dict, idx):
        """Auxiliary loss: squared error of the predicted change of the future frames' features
        from the current frame, in feature-std units, averaged over the samples that have the
        future frame. Also returns the explained share of the change (1 - error / change^2)."""
        t = self.torch
        fut, cur = data["fut_rows"][idx], data["rows"][idx][:, -1]
        ok = (fut >= 0).float()
        fut = fut.clamp(min=0)
        p_cls, p_grid = self.model.predict_future(memory, mask, data["target"][idx])
        err = base = 0.0
        for feat, pred, std in ((self.cls, p_cls, self.stats["cls"][1]),
                                (self.grid, p_grid, self.stats["grid"][1] if self.grid is not None else None)):
            if pred is None:
                continue
            delta = (feat[fut].float() - feat[cur][:, None].float()) / std
            dims = tuple(range(2, delta.dim()))
            err = err + ((pred.float() - delta) ** 2).mean(dims)
            base = base + (delta ** 2).mean(dims)
        n = ok.sum().clamp(min=1.0)
        err, base = (err * ok).sum() / n, (base * ok).sum() / n
        return err, {"fut": err.detach(), "fut_r2": (1 - err / base.clamp(min=1e-6)).detach()}

    def _feature_stats(self, n: int = 5000) -> dict:
        """Per-dimension mean/std of the CLS and grid features, on a fixed subsample."""
        t = self.torch
        sel = t.randperm(len(self.cls), generator=t.Generator().manual_seed(0))[:n].to(self.device)
        out = {"cls": (self.cls[sel].float().mean(0), self.cls[sel].float().std(0))}
        if self.grid is not None:
            g = self.grid[sel].float().flatten(0, -2)
            out["grid"] = (g.mean(0), g.std(0))
        return out

    def _augment(self, cls, grid, aug: Augment):
        """cls (B, HIST, D), grid (B, HIST, cells, D) or None; the current frame is the last.
        Returns the noised features and the keep masks of frames (B, HIST) and grid cells
        (B, HIST, cells) or None; dropped tokens are masked out of the attention."""
        t = self.torch
        b, h = cls.shape[:2]
        if aug.noise:
            cls = cls + t.randn_like(cls) * self.stats["cls"][1] * aug.noise
            if grid is not None:
                grid = grid + t.randn_like(grid) * self.stats["grid"][1] * aug.noise
        keep = t.ones(b, h, dtype=t.bool, device=cls.device)
        if aug.frame_drop:
            keep[:, :-1] = t.rand(b, h - 1, device=cls.device) >= aug.frame_drop
        if aug.vision_drop:
            keep = keep & (t.rand(b, 1, device=cls.device) >= aug.vision_drop)
        cells = None
        if grid is not None and aug.token_drop:
            cells = t.rand(grid.shape[:-1], device=grid.device) >= aug.token_drop
        return cls, grid, keep, cells

    def predict(self, s: pd.DataFrame, batch: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        t = self.torch
        # attention over ~245 vision tokens: a 1024 batch needs ~2 GB per layer, too much on CPU
        batch = batch or (1024 if self.device.startswith("cuda") else 128)
        data = self.tensors(s)
        self.model.eval()
        trajs, probs = [], []
        with t.no_grad(), self._autocast():
            for i in range(0, len(s), batch):
                tr, lg = self.forward(data, slice(i, i + batch))
                trajs.append(tr.float().cpu().numpy())
                probs.append(lg.float().softmax(-1).cpu().numpy())
        return np.concatenate(trajs), np.concatenate(probs)

    def _autocast(self):
        t = self.torch
        if self.device.startswith("cuda"):
            return t.autocast("cuda", dtype=t.float16)
        return t.autocast("cpu", enabled=False)


# ---------------------------------------------------------------- metrics

def multimodal_metrics(traj: np.ndarray, probs: np.ndarray, gt: np.ndarray, miss: float = 2.0) -> pd.DataFrame:
    """Per-sample metrics. traj (M, K, FUT, 3), probs (M, K), gt (M, FUT, 3)."""
    top = probs.argmax(axis=1)
    m = displacement_metrics(traj[np.arange(len(top)), top], gt, miss)
    err = np.linalg.norm(traj[..., :2] - gt[:, None, :, :2], axis=-1)  # (M, K, FUT)
    ade_k, fde_k = err.mean(-1), err[..., -1]
    end = traj[:, :, -1, :2]
    spread = np.linalg.norm(end[:, :, None] - end[:, None], axis=-1).mean(axis=(1, 2))
    return pd.DataFrame({
        **{k: m[k] for k in TOP1},
        "min_ade": ade_k.min(1), "min_fde": fde_k.min(1), "min_miss": fde_k.min(1) > miss,
        "spread": spread,
        "p_top": probs.max(1), "top_is_best": top == ade_k.argmin(1),
    })


def calibration(m: pd.DataFrame, bins: int = 10) -> dict:
    """Expected calibration error of the top-mode probability against 'the top mode is the
    closest one', and the mean confidence/accuracy."""
    b = np.minimum((m["p_top"] * bins).astype(int), bins - 1)
    g = m.groupby(b).agg(conf=("p_top", "mean"), acc=("top_is_best", "mean"), n=("p_top", "size"))
    ece = float((g["n"] * (g["conf"] - g["acc"]).abs()).sum() / len(m))
    return {"ece": ece, "confidence": float(m["p_top"].mean()), "top_is_best": float(m["top_is_best"].mean())}


def evaluate(runner: Runner, s: pd.DataFrame, label: str) -> dict:
    traj, probs = runner.predict(s)
    m = multimodal_metrics(traj, probs, future(s))
    model_cols = [f"{label}_{c}" for c in TOP1 + ["min_ade", "min_fde", "min_miss", "spread"]]
    per = m[TOP1 + ["min_ade", "min_fde", "min_miss", "spread"]].set_axis(model_cols, axis=1)
    table = stratified(s, pd.concat([per, b0_metrics(s)], axis=1))
    return {"by_scenario": table, "calibration": calibration(m), "per_sample": m}


# ---------------------------------------------------------------- training

def _lr_lambda(warmup: int, total: int):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return f


def fit(runner: Runner, train: pd.DataFrame, val: pd.DataFrame, cfg: TrainConfig, out: Path,
        eval_every: int = 1) -> list[dict]:
    """Train with AdamW + warmup/cosine, AMP on CUDA; keep the checkpoint with the best val
    top-1 ADE. Returns the per-epoch history (also written to out/history.json)."""
    from .model import wta_loss

    t = runner.torch
    t.manual_seed(cfg.seed)
    model = runner.model
    data = runner.tensors(train)
    aux = bool(runner.mcfg.future) and runner.cls is not None
    if aux:
        data["fut_rows"] = runner.future_rows(train)
    anchors = None
    if runner.mcfg.anchors:
        from .model import fit_anchors

        model.set_anchors(fit_anchors(future(train), runner.mcfg.k, seed=cfg.seed))
        anchors = model.anchors.float()
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train) / cfg.batch)
    sched = t.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(min(cfg.warmup_steps, steps_per_epoch * cfg.epochs // 10 + 1),
                                                          steps_per_epoch * cfg.epochs))
    cuda = runner.device.startswith("cuda")
    scaler = t.amp.GradScaler("cuda", enabled=cuda)
    gen = t.Generator(device=runner.device).manual_seed(cfg.seed)
    history, best = [], math.inf
    start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        perm = t.randperm(len(train), device=runner.device, generator=gen)
        sums = {}
        for i in range(steps_per_epoch):
            idx = perm[i * cfg.batch:(i + 1) * cfg.batch]
            with runner._autocast():
                if aux:
                    traj, logits, memory, mask = runner.forward(data, idx, cfg.aug, memory=True)
                else:
                    traj, logits = runner.forward(data, idx, cfg.aug)
            loss, parts = wta_loss(traj.float(), logits, data["target"][idx], runner.mcfg, anchors)
            if aux:
                with runner._autocast():
                    fut_loss, fut_parts = runner.future_loss(memory, mask, data, idx)
                loss = loss + runner.mcfg.future_weight * fut_loss
                parts.update(fut_parts)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in {"loss": loss.detach(), **parts}.items():
                sums[k] = sums.get(k, 0.0) + v
        if cuda:
            t.cuda.synchronize()
        row = {"epoch": epoch, **{k: float(v) / steps_per_epoch for k, v in sums.items()},
               "seconds": time.time() - t0, "samples_per_s": len(train) / (time.time() - t0)}
        if len(val) and (epoch % eval_every == 0 or epoch == cfg.epochs):
            traj, probs = runner.predict(val)
            vm = multimodal_metrics(traj, probs, future(val))
            row.update({f"val_{k}": float(vm[k].mean()) for k in ("ade", "fde", "min_ade", "min_fde", "spread")})
            if row["val_ade"] < best:
                best = row["val_ade"]
                save(runner, cfg, out / "best.pt")
        history.append(row)
        log.info("%s epoch %d: %s", cfg.name, epoch, {k: round(v, 4) for k, v in row.items() if k != "epoch"})
        (out / "history.json").write_text(json.dumps(history, indent=1))
        if cfg.max_minutes and time.time() - start > cfg.max_minutes * 60:
            log.warning("%s: stopping after %.1f min (max_minutes)", cfg.name, (time.time() - start) / 60)
            break
    save(runner, cfg, out / "last.pt")
    return history


def save(runner: Runner, cfg: TrainConfig, path: Path) -> None:
    t = runner.torch
    tmp = path.with_name(path.name + ".partial")
    t.save({"model": runner.model.state_dict(), "config": {**asdict(cfg), "model": cfg.model.dict()},
            "kin_columns": runner.cols, "kin_norm": runner.norm.dict()}, tmp)
    tmp.replace(path)


def load_weights(runner: Runner, path: Path) -> None:
    state = runner.torch.load(path, map_location=runner.device, weights_only=False)
    runner.model.load_state_dict(state["model"])


# ---------------------------------------------------------------- entry points

def run(cfg: TrainConfig, frames_for, device: str = "cuda") -> dict:
    """Train one variant and evaluate its best checkpoint on val and test. `frames_for(tables)`
    returns the Frames covering those sample tables (real features, or fake for G3)."""
    out = paths.OUT / "runs" / cfg.name
    smp = load_samples(cfg.train_table)
    train = select(smp, "train", cfg.fraction, drop_flagged=True)
    if cfg.trivial_keep < 1.0:
        n = len(train)
        train = curate(train, cfg.trivial_keep, cfg.trivial_ade)
        log.info("curation: %d of %d train samples kept (trivial: B0-CTRV ADE < %.2f m, %.0f%% kept)",
                 len(train), n, cfg.trivial_ade, 100 * cfg.trivial_keep)
    val = select(smp, "val")
    test = load_samples(cfg.test_table) if cfg.test_table else smp.iloc[:0]
    cols = kin_columns(cfg.model.variant)
    norm = Normalizer.fit(train[cols].to_numpy(np.float64))
    _, vision = VARIANTS[cfg.model.variant]
    tables = [train, val, test]
    if vision and cfg.model.future:
        pool = smp[smp["subset"] == "train"]  # also the QC-flagged samples: their frames can be future frames
        train = train.assign(**future_frames(train, pool, cfg.model.future))
        tables.append(pool)
        log.info("future frames at %s s: %s of the train samples", cfg.model.future,
                 ", ".join("%.0f%%" % (100 * train[f"fut_f0_{i}"].notna().mean()) for i in range(len(cfg.model.future))))
    frames = frames_for(tables) if vision else Frames.empty()
    runner = Runner(cfg.model, frames, norm, device, cfg.seed)
    n_params = sum(p.numel() for p in runner.model.parameters())
    log.info("%s: variant %s, %d train / %d val / %d test samples, %.2fM parameters",
             cfg.name, cfg.model.variant, len(train), len(val), len(test), n_params / 1e6)
    history = fit(runner, train, val, cfg, out)
    if (out / "best.pt").exists():
        load_weights(runner, out / "best.pt")
    result = {"config": {**asdict(cfg), "model": cfg.model.dict()}, "parameters": n_params,
              "train_samples": len(train), "history": history}
    md = [f"# {cfg.name} (variant {cfg.model.variant})", ""]
    for label, part in (("val", val), ("test", test)):
        if len(part) == 0:
            continue
        ev = evaluate(runner, part, cfg.model.variant)
        result[label] = {"by_scenario": ev["by_scenario"].to_dict(orient="records"), "calibration": ev["calibration"]}
        ev["per_sample"].assign(sample_id=part["sample_id"].to_numpy()).to_parquet(out / f"per_sample_{label}.parquet")
        md += [f"## {label}", "", md_table(ev["by_scenario"]), "", f"calibration: {ev['calibration']}", ""]
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    (out / "result.md").write_text("\n".join(md))
    return result


def export_predictions(name: str, frames_for, device: str = "cuda") -> dict:
    """K trajectories + mode probabilities of a run's best checkpoint for every val and test
    sample -> outputs/runs/NAME/predictions_{val,test}.npz (sample_id, traj (M, K, FUT, 3),
    probs (M, K), float16), for the viewer and the README figures without a GPU."""
    import torch

    out = paths.OUT / "runs" / name
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    cfg = state["config"]
    mcfg = ModelConfig(**cfg["model"])
    norm = Normalizer(np.asarray(state["kin_norm"]["mean"]), np.asarray(state["kin_norm"]["std"]))
    smp = load_samples(cfg["train_table"])
    parts = {"val": select(smp, "val"), "test": load_samples(cfg["test_table"]) if cfg["test_table"] else smp.iloc[:0]}
    _, vision = VARIANTS[mcfg.variant]
    frames = frames_for(list(parts.values())) if vision else Frames.empty()
    runner = Runner(mcfg, frames, norm, device)
    runner.model.load_state_dict(state["model"])
    counts = {}
    for label, part in parts.items():
        if len(part) == 0:
            continue
        traj, probs = runner.predict(part)
        np.savez(out / f"predictions_{label}.npz", sample_id=part["sample_id"].to_numpy().astype(str),
                 traj=traj.astype(np.float16), probs=probs.astype(np.float16))
        counts[label] = len(part)
    return counts


def compare(names: list[str], split: str = "val") -> str:
    """Variants side by side by scenario (top-1 ADE/FDE/miss, minADE), with B0-CTRV/CV."""
    rows = None
    for name in names:
        r = json.loads((paths.OUT / "runs" / name / "result.json").read_text())
        v = r["config"]["model"]["variant"]
        t = pd.DataFrame(r[split]["by_scenario"]).set_index("scenario")
        cols = {f"{v}_ade": f"{name} ADE", f"{v}_fde": f"{name} FDE", f"{v}_miss": f"{name} miss%",
                f"{v}_min_ade": f"{name} minADE"}
        part = t[list(cols)].rename(columns=cols)
        if rows is None:
            rows = t[["n", "ctrv_ade", "ctrv_fde", "ctrv_miss", "cv_ade"]].rename(
                columns={"ctrv_ade": "B0-CTRV ADE", "ctrv_fde": "B0-CTRV FDE", "ctrv_miss": "B0-CTRV miss%",
                         "cv_ade": "B0-CV ADE"})
        rows = rows.join(part)
    table = md_table(rows.reset_index())
    out = paths.OUT / "runs" / f"compare_{split}_{'_'.join(names)}.md"
    out.write_text(f"# {split}: {', '.join(names)}\n\nMetres over 4 s; miss = FDE > 2 m (%).\n\n{table}\n")
    return table


def g3(n: int = 1000, epochs: int = 200, device: str = "cpu", model: ModelConfig | None = None,
       frames: str = "fake", target_ade: float = 0.5, batch: int = 32) -> dict:
    """Gate G3: overfit ~n train samples (variant C). Passes when the loss drops by >= 98% and
    the training minADE falls below target_ade metres; otherwise there is a bug. Fake features are unique per frame, so the
    model can memorise the samples; real features (on a GPU host) repeat the check for real."""
    mcfg = model or ModelConfig(variant="C", dropout=0.0)
    cfg = TrainConfig(name="g3", model=mcfg, test_table="", epochs=epochs, batch=min(batch, n), lr=1e-3,
                      weight_decay=0.0, warmup_steps=20)
    smp = load_samples(cfg.train_table)
    sub = select(smp, "train", drop_flagged=True).sample(n=n, random_state=0).reset_index(drop=True)
    if frames == "fake":
        fr = Frames.fake(hist_paths(sub))
    else:
        from .features import DEFAULT, FeatureStore
        from .manifest import Manifest

        fr = Frames.from_store(FeatureStore(DEFAULT, Manifest(paths.MANIFEST)), hist_paths(sub))
    cols = kin_columns(mcfg.variant)
    runner = Runner(mcfg, fr, Normalizer.fit(sub[cols].to_numpy(np.float64)), device, cfg.seed)
    out = paths.OUT / "g3"
    t0 = time.time()
    history = fit(runner, sub, sub.iloc[:0], cfg, out)
    traj, probs = runner.predict(sub)
    m = multimodal_metrics(traj, probs, future(sub))
    result = {"samples": n, "epochs": len(history), "minutes": (time.time() - t0) / 60, "frames": frames,
              "first_loss": history[0]["loss"], "last_loss": history[-1]["loss"],
              "train_min_ade": float(m["min_ade"].mean()), "train_top1_ade": float(m["ade"].mean()),
              "target_min_ade": target_ade}
    result["passed"] = bool(result["last_loss"] < 0.02 * result["first_loss"] and result["train_min_ade"] < target_ade)
    (out / "report.json").write_text(json.dumps(result, indent=1))
    log.info("G3: %s", result)
    return result
