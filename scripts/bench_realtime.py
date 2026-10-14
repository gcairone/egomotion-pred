"""End-to-end inference latency of the full pipeline for one new frame (steady-state streaming:
the 3 older frames' features are cached): JPEG decode -> DINOv2 -> PCA -> trajectory head.

python scripts/bench_realtime.py --run full_C_anch_lr1e4 --jpeg frame.jpg --device cuda
"""
import argparse
import io
import json
import time

import numpy as np
import torch

from egopred import features
from egopred.model import ModelConfig, TrajectoryModel
from egopred.trajectory import HIST


def stats(ms):
    a = np.asarray(ms)
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full_C_anch_lr1e4")
    ap.add_argument("--jpeg", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--fp16", type=int, default=1, help="backbone autocast on cuda (as in extraction)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = a.device
    cuda = dev.startswith("cuda")
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    cfg = features.DEFAULT
    state = torch.load(f"outputs/runs/{a.run}/best.pt", map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**state["config"]["model"])
    kin_dim = len(state["kin_columns"]) + 4
    pca = np.load(features.config_dir(cfg) / "pca.npz")
    mean = torch.from_numpy(pca["mean"]).to(dev)
    comps = torch.from_numpy(pca["components"]).to(dev)
    head = TrajectoryModel(mcfg, kin_dim, 768, cfg.pca_dim, cfg.grid[0] * cfg.grid[1]).to(dev).eval()
    head.load_state_dict(state["model"])
    fwd, _ = features.build_backbone(cfg, dev)
    data = open(a.jpeg, "rb").read()

    cache_cls = torch.randn(HIST - 1, 768, device=dev)
    cache_grid = torch.randn(HIST - 1, 60, cfg.pca_dim, device=dev)
    kin = torch.randn(1, kin_dim, device=dev)

    t_dec, t_bb, t_pca, t_head, t_all = [], [], [], [], []
    with torch.inference_mode():
        for i in range(a.warmup + a.n):
            t0 = time.perf_counter()
            img = torch.from_numpy(features.decode(data, cfg.input_hw))[None]
            t1 = time.perf_counter()
            cls, grid = fwd(img)
            sync()
            t2 = time.perf_counter()
            g = ((grid.reshape(1, 60, -1) - mean) @ comps.T)
            sync()
            t3 = time.perf_counter()
            c_all = torch.cat([cache_cls, cls])[None]
            g_all = torch.cat([cache_grid, g])[None]
            traj, logits = head(kin, c_all, g_all)
            traj = traj.float().cpu().numpy()  # result back on the host
            t4 = time.perf_counter()
            if i >= a.warmup:
                t_dec.append((t1 - t0) * 1e3); t_bb.append((t2 - t1) * 1e3)
                t_pca.append((t3 - t2) * 1e3); t_head.append((t4 - t3) * 1e3)
                t_all.append((t4 - t0) * 1e3)
    res = {"device": dev, "gpu": torch.cuda.get_device_name() if cuda else "cpu",
           "threads": torch.get_num_threads(), "n": a.n, "run": a.run,
           "decode": stats(t_dec), "backbone": stats(t_bb), "pca": stats(t_pca),
           "head": stats(t_head), "total": stats(t_all),
           "fps_serial": 1e3 / float(np.mean(t_all))}
    # backbone-only batched throughput (offline-style upper bound)
    if cuda:
        x = torch.randint(0, 255, (32, 3, *cfg.input_hw), dtype=torch.uint8)
        with torch.inference_mode():
            fwd(x[:4]); sync(); t = time.perf_counter()
            for _ in range(5):
                fwd(x)
            sync()
        res["backbone_batch32_fps"] = 5 * 32 / (time.perf_counter() - t)
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
