"""Gate G0: video/pose sync and ground-truth projection on comma2k19 and NAVSIM mini.

Outputs under outputs/g0/<dataset>/: overlay PNGs (GT ribbon in green, B0 baselines) and
report.json with the sync check and a vanishing-point convergence check.
"""

import io
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from . import comma2k19, paths
from .geometry import make_T, step_yaw, wrap_angle
from .sync import sync_summary, visual_yaw
from .trajectory import DT, constant_turn_rate, constant_velocity, tag_scenarios, valid_refs, windows
from .viz import bev, overlay, side_by_side

log = logging.getLogger(__name__)

GREEN, RED, ORANGE = (40, 230, 60), (240, 60, 60), (255, 170, 0)
STRAIGHT_DEG = 2.0  # |heading change| over 4 s below which a sample counts as straight
MIN_SPEED_VP = 5.0  # m/s, minimum speed for the vanishing-point check


def convergence_check(w: dict, speed: np.ndarray, T_ego_cam, K, dist) -> dict:
    """On straight, moving samples the 4 s waypoint must project close to the image column of
    the forward direction (the vanishing point) and below it."""
    from .geometry import inv_T, project, transform_points

    sel = (np.abs(np.degrees(wrap_angle(w["fut"][:, -1, 2]))) < STRAIGHT_DEG) & (speed > MIN_SPEED_VP)
    if sel.sum() == 0:
        return {"n": 0}
    T_cam_ego = inv_T(T_ego_cam)
    far = np.c_[w["fut"][sel, -1, :2], w["fut_z"][sel, -1]]
    uv, ok = project(transform_points(T_cam_ego, far), K, dist)
    # vanishing point of the ego forward axis
    fwd = T_cam_ego[:3, :3] @ np.array([1.0, 0.0, 0.0])
    vp, _ = project(fwd[None] * 1000.0, K, dist)
    du = np.abs(uv[ok, 0] - vp[0, 0])
    return {
        "n": int(sel.sum()),
        "vanishing_point_px": vp[0].round(1).tolist(),
        "median_abs_du_px": float(np.median(du)),
        "p90_abs_du_px": float(np.percentile(du, 90)),
        "frac_below_vp": float((uv[ok, 1] > vp[0, 1]).mean()),
        "image_width": None,
    }


def pick_examples(tags: dict, speed: np.ndarray, n_each: int = 1) -> dict[str, int]:
    """Indices of illustrative samples: strongest left/right turns, a lane change, a fast straight."""
    dpsi = tags["heading_change_deg"]
    out = {}
    if dpsi.max() > 5:
        out["left"] = int(np.argmax(dpsi))
    if dpsi.min() < -5:
        out["right"] = int(np.argmin(dpsi))
    lc = np.flatnonzero(tags["flags"]["lateral_shift"])
    if len(lc):
        out["lateral_shift"] = int(lc[np.argmax(np.abs(tags["lateral_end"][lc]))])
    straight = np.flatnonzero((np.abs(dpsi) < STRAIGHT_DEG) & (speed > MIN_SPEED_VP))
    if len(straight):
        out["straight"] = int(straight[np.argmax(speed[straight])])
    return out


def render_example(img: Image.Image, w: dict, j: int, speed, yaw_rate, v_ego, T_ego_cam, K, dist,
                   title: str) -> tuple[Image.Image, dict]:
    gt = w["fut"][j]
    cv = constant_velocity(v_ego[j:j + 1])[0]
    ctrv = constant_turn_rate(speed[j:j + 1], yaw_rate[j:j + 1])[0]
    im, vis = overlay(img, [(gt, w["fut_z"][j], GREEN, "gt"), (cv, None, RED, "b0_cv"),
                            (ctrv, None, ORANGE, "b0_ctrv")], T_ego_cam, K, dist, text=title)
    b = bev([(gt, GREEN, "gt"), (cv, RED, "cv"), (ctrv, ORANGE, "ctrv")], size=im.height)
    return side_by_side(im, b), vis


def run_comma(out_dir: Path | None = None) -> dict:
    out_dir = out_dir or paths.OUT / "g0" / "comma2k19"
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = comma2k19.fetch_segment()
    t, T, T_ego_cam, calib = comma2k19.load_poses(seg)
    every = int(round(DT * comma2k19.FPS))
    # half-resolution grayscale for visual odometry; the bonnet (bottom rows) is masked out
    half_K = comma2k19.K * np.array([[0.5], [0.5], [1.0]])
    grays, keep = [], {}
    for i, rgb in comma2k19.iter_frames(seg):
        grays.append(np.ascontiguousarray(np.asarray(Image.fromarray(rgb).convert("L").reduce(2))))
        if i % every == 0:
            keep[i] = rgb
    if len(grays) != len(t):
        raise RuntimeError(f"{len(grays)} video frames vs {len(t)} poses")
    mask = np.zeros_like(grays[0])
    mask[: int(comma2k19.BONNET_ROW / 2)] = 255

    # sync: yaw over 0.5 s pairs (i, i + 10) for every frame i, lags in 20 Hz frames
    R_ego_cam = T_ego_cam[:3, :3]
    img_dyaw = np.array([visual_yaw(grays[i], grays[i + every], half_K, None, R_ego_cam, mask)[0]
                         for i in range(len(grays) - every)])
    pose_dyaw = step_yaw(T, every)
    sync = sync_summary(img_dyaw, pose_dyaw, lags=range(-every, every + 1), lag_tol=2)
    sync["lag_unit_s"] = 1 / comma2k19.FPS

    # 2 Hz windows, as for NAVSIM
    sub = np.arange(0, len(t), every)
    t2, T2 = t[sub], T[sub]
    refs = valid_refs(len(sub))
    w = windows(t2, T2, refs)
    speed, yaw_rate = w["speed_pose"], w["yaw_rate"]
    v_ego = np.c_[speed, np.zeros_like(speed)]
    tags = tag_scenarios(w["fut"], speed)
    conv = convergence_check(w, speed, T_ego_cam, comma2k19.K, None)
    conv["image_width"] = comma2k19.IMAGE_SIZE[0]

    shots = {}
    for name, j in pick_examples(tags, speed).items():
        frame = int(sub[refs[j]])
        im, vis = render_example(Image.fromarray(keep[frame]), w, j, speed, yaw_rate, v_ego,
                                 T_ego_cam, comma2k19.K, None,
                                 f"comma2k19 frame {frame} {name} v={speed[j]:.1f}m/s "
                                 f"dpsi={tags['heading_change_deg'][j]:.1f}deg")
        fn = out_dir / f"overlay_{name}.png"
        im.save(fn)
        shots[name] = {"frame": frame, "file": fn.name, "gt_visible": vis["gt"]}

    report = {
        "dataset": "comma2k19", "segment": comma2k19.EXAMPLE_SEGMENT, "frames": len(t),
        "mount_calibration": calib, "windows_2hz": len(refs), "gaps": int(w["gap"].sum()), "pose_jumps": int(w["pose_jump"].sum()),
        "sync_20hz": sync, "convergence": conv, "examples": shots,
        "scenarios": {k: int(v.sum()) for k, v in tags["flags"].items()},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def openscene_camera(df) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K, distortion and T_ego_cam of CAM_F0 for one log (asserting they are constant)."""
    from .openscene.metadata import array

    K = array(df, "f0K", (3, 3))
    dist = array(df, "f0dist")
    T_ego_cam = array(df, "l2e", (4, 4)) @ make_T(array(df, "f0R", (3, 3)), array(df, "f0t"))
    for a in (K, dist, T_ego_cam):
        if not np.allclose(a, a[0], atol=1e-6):
            raise ValueError("CAM_F0 calibration changes within a log")
    return K[0], dist[0], T_ego_cam[0]


def run_navsim(split: str, archive: str, out_dir: Path | None = None) -> dict:
    import pandas as pd

    from .openscene.frames import shard_paths
    from .openscene.metadata import array, load_log

    out_dir = out_dir or paths.OUT / "g0" / f"navsim_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path, index_path = shard_paths(split, archive)
    idx = pd.read_parquet(index_path).set_index("f0_path")
    report = {"dataset": f"openscene/{split}", "archive": archive, "logs": {}}
    f = open(tar_path, "rb")

    def read_img(rel: str, draft=None) -> Image.Image:
        row = idx.loc[rel]
        f.seek(int(row["offset"]))
        im = Image.open(io.BytesIO(f.read(int(row["size"]))))
        if draft:
            im.draft(*draft)
        return im

    for log_name in sorted(idx["log_name"].unique()):
        df = load_log(split, log_name)
        df = df[df["f0_path"].isin(idx.index)].reset_index(drop=True)
        K, dist, T_ego_cam = openscene_camera(df)
        t = df["timestamp"].to_numpy() / 1e6
        T = array(df, "e2g", (4, 4))

        # sync at the native 2 Hz on consecutive frames, half-resolution (DCT draft) decodes
        half_K = K * np.array([[0.5], [0.5], [1.0]])
        img_dyaw = np.full(len(df) - 1, np.nan)
        prev = None
        for i, rel in enumerate(df["f0_path"]):
            im = read_img(rel, ("L", (960, 540)))
            if im.size != (960, 540):
                raise ValueError(f"unexpected draft size {im.size}")
            cur = np.asarray(im.convert("L"))
            if prev is not None:
                img_dyaw[i - 1] = visual_yaw(prev, cur, half_K, dist, T_ego_cam[:3, :3])[0]
            prev = cur
        pose_dyaw = step_yaw(T)
        img_dyaw[np.abs(np.diff(t) - DT) > 0.1] = np.nan  # pairs across a gap
        sync = sync_summary(img_dyaw, pose_dyaw, lags=range(-2, 3))
        sync["lag_unit_s"] = DT

        refs = valid_refs(len(df))
        w = windows(t, T, refs)
        dyn = array(df, "dyn")[refs]
        v_ego = dyn[:, :2]
        speed_r = np.hypot(dyn[:, 0], dyn[:, 1])
        tags = tag_scenarios(w["fut"], speed_r)
        conv = convergence_check(w, speed_r, T_ego_cam, K, dist)
        conv["image_width"] = 1920

        shots = {}
        for name, j in pick_examples(tags, speed_r).items():
            rel = df["f0_path"].iloc[refs[j]]
            im, vis = render_example(read_img(rel), w, j, speed_r, w["yaw_rate"], v_ego, T_ego_cam, K, dist,
                                     f"{log_name[:30]} {name} v={speed_r[j]:.1f}m/s "
                                     f"dpsi={tags['heading_change_deg'][j]:.1f}deg")
            fn = out_dir / f"{log_name}_{name}.png"
            im.save(fn)
            shots[name] = {"token": df["token"].iloc[refs[j]], "file": fn.name, "gt_visible": vis["gt"]}
        report["logs"][log_name] = {
            "frames": len(df), "windows": len(refs), "gaps": int(w["gap"].sum()),
            "pose_jumps": int(w["pose_jump"].sum()), "sync_2hz": sync, "convergence": conv,
            "speed_pose_vs_dyn_median_abs": float(np.median(np.abs(w["speed_pose"] - speed_r))),
            "camera": {"K": K.round(2).tolist(), "dist": dist.round(4).tolist(),
                       "T_ego_cam": T_ego_cam.round(4).tolist()},
            "examples": shots,
            "scenarios": {k: int(v.sum()) for k, v in tags["flags"].items()},
        }
    f.close()
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report
