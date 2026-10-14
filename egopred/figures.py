"""README figures: camera frame with the predicted paths of two runs next to a perspective
view of the same paths on a ground grid. Uses the exported predictions, no GPU needed."""

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import paths
from .g0 import openscene_camera
from .openscene.frames import read_frame
from .openscene.metadata import load_log
from .openscene.samples import future, load_samples
from .viz import _font, overlay, perspective_view

log = logging.getLogger(__name__)

GT = (90, 205, 120)
BLUE = (70, 140, 255)
ORANGE = (255, 150, 30)
PANEL = (800, 450)
NATIVE_WIDTH = 1920  # width the OpenScene calibration refers to; stored frames are downscaled


def load_pred(run: str, split: str) -> pd.DataFrame:
    z = np.load(paths.OUT / "runs" / run / f"predictions_{split}.npz")
    top = z["probs"].astype(np.float32).argmax(1)
    traj = z["traj"][np.arange(len(top)), top].astype(np.float64)
    ps = pd.read_parquet(paths.OUT / "runs" / run / f"per_sample_{split}.parquet", columns=["sample_id", "ade", "fde"])
    df = pd.DataFrame({"sample_id": z["sample_id"].astype(str), "row": np.arange(len(top))})
    return df.merge(ps, on="sample_id").set_index("sample_id").assign(traj=lambda d: list(traj[d["row"]]))


def pick_sample(a: pd.DataFrame, c: pd.DataFrame, smp: pd.DataFrame, scenario: str | None, pick: str,
                min_speed: float = 3.0, seed: int = 0) -> str:
    """gain: C beats A the most; loss: A beats C the most; median: typical C-A difference at
    a mid-range ADE; random: seed-controlled."""
    t = smp.set_index("sample_id")
    ids = a.index.intersection(c.index).intersection(t.index)
    t = t.loc[ids]
    ok = (t["speed"] > min_speed) & ~t["gap"] & ~t["pose_jump"]
    if scenario:
        ok &= t["scenario"] == scenario
    ids = t.index[ok]
    diff = (a.loc[ids, "ade"] - c.loc[ids, "ade"]).to_numpy()
    if pick == "gain":
        return str(ids[int(np.argmax(diff))])
    if pick == "loss":
        return str(ids[int(np.argmin(diff))])
    if pick == "random":
        return str(ids[int(np.random.default_rng(seed).integers(len(ids)))])
    mid = c.loc[ids, "ade"].to_numpy()
    near = np.argsort(np.abs(mid - np.median(mid)))[:50]
    return str(ids[near[int(np.argmax(diff[near]))]])


class FrameStore:
    """CAM_F0 frames of one split by (log, timestamp), with per-log metadata cached."""

    def __init__(self, split: str):
        self.split = split
        self.logs: dict[str, tuple[pd.DataFrame, tuple]] = {}
        parts = []
        for ix in sorted((paths.OPENSCENE_FRAMES / split).glob("*.index.parquet")):
            df = pd.read_parquet(ix)
            df["tar"] = str(ix.with_name(ix.name.replace(".index.parquet", ".tar")))
            parts.append(df)
        self.index = pd.concat(parts).set_index("f0_path")

    def get(self, log_name: str, timestamp: int):
        """Decoded frame plus the camera (K scaled to the stored size, dist, T_ego_cam) of its log."""
        if log_name not in self.logs:
            df = load_log(self.split, log_name)
            self.logs[log_name] = (df.set_index("timestamp")["f0_path"], openscene_camera(df))
        rel_by_ts, (K, dist, T_ego_cam) = self.logs[log_name]
        row = self.index.loc[rel_by_ts.loc[timestamp]]
        data = read_frame(Path(row["tar"]), int(row["offset"]), int(row["size"]))
        img = Image.open(io.BytesIO(data)).convert("RGB")
        K = K * np.array([[img.width / NATIVE_WIDTH], [img.width / NATIVE_WIDTH], [1.0]])
        return img, K, dist, T_ego_cam


def compose(img, K, dist, T_ego_cam, gt, pa, pc, name_a: str, name_c: str, ade_a: float, ade_c: float,
            title: str, panel=PANEL) -> Image.Image:
    left, _ = overlay(img, [(gt, None, GT, "gt"), (pa, None, BLUE, name_a), (pc, None, ORANGE, name_c)],
                      T_ego_cam, K, dist, out_width=panel[0], text=title)
    left = left.resize(panel)
    right = perspective_view([(gt, GT, "gt"), (pa, BLUE, name_a), (pc, ORANGE, name_c)], size=panel)
    out = Image.new("RGB", (panel[0] * 2, panel[1] + 44), (18, 20, 24))
    out.paste(left, (0, 0))
    out.paste(right, (panel[0], 0))
    d = ImageDraw.Draw(out)
    f = _font(20)
    x = 16
    for label, col, extra in [("ground truth", GT, ""), (name_a, BLUE, f"  ADE {ade_a:.2f} m"),
                              (name_c, ORANGE, f"  ADE {ade_c:.2f} m")]:
        d.ellipse([x, panel[1] + 14, x + 16, panel[1] + 30], fill=col)
        d.text((x + 24, panel[1] + 22), label + extra, fill=(235, 238, 242), font=f, anchor="lm")
        x += 24 + int(d.textlength(label + extra, font=f)) + 36
    return out


def make_figure(run_a: str, run_c: str, out: Path, split: str = "test", table: str = "navtest", frames_split: str = "test",
                sample: str | None = None, scenario: str | None = None, pick: str = "median",
                seed: int = 0) -> dict:
    a, c = load_pred(run_a, split), load_pred(run_c, split)
    smp = load_samples(table)
    sid = sample or pick_sample(a, c, smp, scenario, pick, seed=seed)
    row = smp[smp["sample_id"] == sid].iloc[0]
    gt = future(smp[smp["sample_id"] == sid])[0]
    img, K, dist, T_ego_cam = FrameStore(frames_split).get(row["log_name"], int(row["timestamp"]))
    title = f"{row['scenario']}  v={row['speed']:.1f} m/s  {row['log_name'][:19]}"
    fig = compose(img, K, dist, T_ego_cam, gt, a.loc[sid, "traj"][:, :3], c.loc[sid, "traj"][:, :3], "A", "C",
                  float(a.loc[sid, "ade"]), float(c.loc[sid, "ade"]), title)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out)
    info = {"sample_id": sid, "scenario": row["scenario"], "ade_a": float(a.loc[sid, "ade"]),
            "ade_c": float(c.loc[sid, "ade"]), "file": str(out)}
    log.info("%s", info)
    return info


def pick_window(a: pd.DataFrame, c: pd.DataFrame, smp: pd.DataFrame, length: int, pick: str = "gain",
                min_speed: float = 3.0, seed: int = 0, scenario: str | None = None) -> pd.DataFrame:
    """`length` consecutive samples (0.5 s apart, same log, all moving, no gaps) of the test table.
    gain: the window where C beats A the most on average; loss: where A beats C the most;
    random: seed-controlled.
    With `scenario`, only windows where at least 60% of the samples have that tag."""
    t = smp.merge(pd.DataFrame({"sample_id": a.index, "ade_a": a["ade"].to_numpy()}), on="sample_id")
    t = t.merge(pd.DataFrame({"sample_id": c.index, "ade_c": c["ade"].to_numpy()}), on="sample_id")
    t = t.sort_values(["log_name", "timestamp"]).reset_index(drop=True)
    step = t.groupby("log_name")["timestamp"].diff() / 1e6
    ok = (t["speed"] > min_speed) & ~t["gap"] & ~t["pose_jump"]
    run = (~((step.round(1) == 0.5) & ok) | ~ok).cumsum()
    t["gain"] = t["ade_a"] - t["ade_c"]
    cands = []
    for _, g in t[ok].groupby(run[ok]):
        if len(g) >= length:
            m = g["gain"].rolling(length).mean().to_numpy()
            for i in range(length - 1, len(g)):
                w = g.iloc[i - length + 1:i + 1]
                if scenario is None or (w["scenario"] == scenario).mean() >= 0.6:
                    cands.append((m[i], w))
    if not cands:
        raise ValueError(f"no run of {length} consecutive moving samples (scenario={scenario})")
    if pick == "random":
        return cands[int(np.random.default_rng(seed).integers(len(cands)))][1]
    if pick == "loss":
        return min(cands, key=lambda x: x[0])[1]
    return max(cands, key=lambda x: x[0])[1]


def palette_for(frames: list[Image.Image]) -> Image.Image:
    """One shared 256-colour palette from a few frames, so the colours do not flicker in the GIF."""
    pick = [frames[i] for i in np.linspace(0, len(frames) - 1, min(6, len(frames))).astype(int)]
    sheet = Image.new("RGB", (pick[0].width, pick[0].height * len(pick)))
    for i, f in enumerate(pick):
        sheet.paste(f, (0, i * f.height))
    return sheet.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def make_gif(run_a: str, run_c: str, out: Path, split: str = "test", table: str = "navtest",
             frames_split: str = "test", length: int = 24, pick: str = "gain", seed: int = 0,
             hz: float = 2.0, panel=(640, 360), scenario: str | None = None) -> dict:
    a, c = load_pred(run_a, split), load_pred(run_c, split)
    smp = load_samples(table)
    win = pick_window(a, c, smp, length, pick, seed=seed, scenario=scenario)
    store = FrameStore(frames_split)
    frames = []
    for _, row in win.iterrows():
        sid = row["sample_id"]
        gt = future(smp[smp["sample_id"] == sid])[0]
        img, K, dist, T_ego_cam = store.get(row["log_name"], int(row["timestamp"]))
        title = f"{row['scenario']}  v={row['speed']:.1f} m/s  t={(row['timestamp'] - win['timestamp'].iloc[0]) / 1e6:.1f} s"
        frames.append(compose(img, K, dist, T_ego_cam, gt, a.loc[sid, "traj"][:, :3], c.loc[sid, "traj"][:, :3],
                              "A", "C", float(a.loc[sid, "ade"]), float(c.loc[sid, "ade"]), title, panel=panel))
    pal = palette_for(frames)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(out, save_all=True, append_images=q[1:], duration=round(1000 / hz), loop=0, optimize=False)
    info = {"log": win["log_name"].iloc[0], "first_sample": win["sample_id"].iloc[0], "frames": len(q),
            "mean_ade_a": float(win["ade_a"].mean()), "mean_ade_c": float(win["ade_c"].mean()),
            "file": str(out), "mb": round(out.stat().st_size / 1e6, 2)}
    log.info("%s", info)
    return info
