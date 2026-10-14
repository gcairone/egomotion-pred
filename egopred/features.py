"""Offline visual features: a frozen DINOv2 backbone run once over every stored front frame.

Per frame we keep the CLS token and a pooled patch grid. The 21x37 patch map is reduced to
`grid` cells by adaptive average pooling and its channels are projected with PCA to
`pca_dim`, so the ~170k frames of navtrain + navtest fit in a few GB. The PCA is fitted once,
on a fixed subset of the current frames of navtrain *train* samples (never val/test).

Two stages, so the GPU works while frames are still being ingested:
  1. backbone pass per frame shard -> CLS + pooled grid at full width ("raw", GPU host only);
  2. once every frame of the PCA subset has raw features: fit the PCA, project every shard.

Final output per frame shard, under FEATURES/<config>/<split>/:
  <archive>.cls.npy (n, D) and <archive>.grid.npy (n, gh, gw, pca_dim), float16,
  <archive>.index.parquet (f0_path, row).
Training reads only these files, never the frames.
"""

import hashlib
import io
import json
import logging
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from PIL import Image

from . import paths
from .manifest import Manifest
from .openscene.frames import shard_paths

log = logging.getLogger(__name__)

TASK_SPLITS = ("trainval", "test")  # mini is development data only
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    model: str  # timm model name
    input_hw: tuple[int, int]  # backbone input; multiples of the 14 px patch, ~16:9
    grid: tuple[int, int]  # pooled patch grid (rows, cols)
    pca_dim: int
    pca_frames: int = 4000  # frames used to fit the PCA


DEFAULT = FeatureConfig(
    name="dinov2b_g6x10",
    model="vit_base_patch14_reg4_dinov2.lvd142m",
    input_hw=(294, 518),
    grid=(6, 10),
    pca_dim=256,
)


def config_dir(cfg: FeatureConfig):
    return paths.FEATURES / cfg.name


def feature_paths(cfg: FeatureConfig, split: str, archive: str):
    base = config_dir(cfg) / split / archive
    return (base.with_name(archive + ".cls.npy"), base.with_name(archive + ".grid.npy"),
            base.with_name(archive + ".index.parquet"))


def decode(data: bytes, input_hw: tuple[int, int]) -> np.ndarray:
    """JPEG bytes -> (3, H, W) uint8 at the backbone input size."""
    h, w = input_hw
    im = Image.open(io.BytesIO(data))
    im.draft("RGB", (w, h))  # DCT-domain downscale when the source is >= 2x larger
    im = im.convert("RGB").resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(im).transpose(2, 0, 1).copy()


class ShardFrames:
    """Map-style dataset over the frames of one tar shard (random access via the index)."""

    def __init__(self, items: list[tuple[str, int, int]], input_hw: tuple[int, int]):
        self.items = items  # (tar path, offset, size)
        self.input_hw = input_hw
        self._files: dict = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> np.ndarray:
        tar, offset, size = self.items[i]
        f = self._files.get(tar)
        if f is None:  # opened lazily, so each loader worker has its own handle
            f = self._files[tar] = open(tar, "rb")
        f.seek(offset)
        return decode(f.read(size), self.input_hw)


def build_backbone(cfg: FeatureConfig, device: str, pretrained: bool = True):
    import timm
    import torch

    vit = timm.create_model(cfg.model, pretrained=pretrained, img_size=cfg.input_hw, num_classes=0)
    vit.eval().requires_grad_(False).to(device)
    patch_hw = tuple(vit.patch_embed.grid_size)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1) * 255

    def forward(images_u8):
        """(B, 3, H, W) uint8 -> CLS (B, D), pooled grid (B, gh, gw, D), float32."""
        x = (images_u8.to(device, non_blocking=True).float() - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            tok = vit.forward_features(x)
        tok = tok.float()
        b, _, d = tok.shape
        patches = tok[:, vit.num_prefix_tokens:].transpose(1, 2).reshape(b, d, *patch_hw)
        grid = torch.nn.functional.adaptive_avg_pool2d(patches, cfg.grid).permute(0, 2, 3, 1)
        return tok[:, 0], grid

    return forward, vit.num_features


def _loader(items, cfg, batch, workers):
    import torch

    ds = ShardFrames(items, cfg.input_hw)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, num_workers=workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=False, collate_fn=lambda xs: torch.from_numpy(np.stack(xs)))


def shard_ready(split: str, archive: str) -> bool:
    """Both shard files fully present. On a GPU host the shards arrive by rsync while features
    run: rsync renames a file only once complete, and the size check also catches truncated
    copies made any other way."""
    tar, index = shard_paths(split, archive)
    if not (tar.exists() and index.exists()):
        return False
    idx = pd.read_parquet(index, columns=["offset", "size"])
    return idx.empty or tar.stat().st_size >= int((idx["offset"] + idx["size"]).max())


def frame_shards(manifest: Manifest, splits=TASK_SPLITS, ready_only: bool = True) -> list[tuple[str, str]]:
    """Verified frame shards of `splits`; with ready_only, only those whose files are present."""
    keys = (k.split("/", 1) for k in manifest.keys("frames", "verified"))
    out = [(split, archive) for split, archive in keys if split in splits]
    if ready_only:
        ready = [s for s in out if shard_ready(*s)]
        if len(ready) < len(out):
            log.info("%d/%d frame shards present, the others are not here (yet)", len(ready), len(out))
        out = ready
    return out


def pending_shards(cfg: FeatureConfig, manifest: Manifest, splits=TASK_SPLITS) -> list[str]:
    """Verified frame shards still without final features (not arrived yet, or PCA not fitted)."""
    done = set(manifest.keys(f"features:{cfg.name}", "verified"))
    keys = (f"{split}/{archive}" for split, archive in frame_shards(manifest, splits, ready_only=False))
    return [k for k in keys if k not in done]


def frame_table(manifest: Manifest, splits=TASK_SPLITS) -> pd.DataFrame:
    """Every frame of the verified frame shards: f0_path, split, archive, tar, offset, size.
    A frame stored in more than one shard is listed once (first shard in key order)."""
    parts = []
    for split, archive in frame_shards(manifest, splits):
        tar, index = shard_paths(split, archive)
        idx = pd.read_parquet(index, columns=["f0_path", "offset", "size"])
        parts.append(idx.assign(split=split, archive=archive, tar=str(tar)))
    if not parts:
        return pd.DataFrame(columns=["f0_path", "offset", "size", "split", "archive", "tar"])
    return pd.concat(parts, ignore_index=True).drop_duplicates("f0_path", ignore_index=True)


def pca_subset(train_paths, n: int) -> list[str]:
    """Deterministic pseudo-random subset: the n paths with the smallest md5."""
    return sorted(set(train_paths), key=lambda p: hashlib.md5(p.encode()).hexdigest())[:n]


def fit_pca(grid_vectors, dim: int) -> dict:
    """PCA of (N, D) vectors (torch); returns mean (D,), components (dim, D), explained ratio."""
    import torch

    x = grid_vectors.double()
    mean = x.mean(0)
    xc = x - mean
    cov = xc.T @ xc / max(len(x) - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    ratio = (evals[:dim].sum() / evals.sum()).item()
    return {"mean": mean.float().cpu().numpy(), "components": evecs[:, :dim].T.float().cpu().numpy(),
            "explained": ratio}


def raw_paths(cfg: FeatureConfig, split: str, archive: str):
    """Stage-1 output: CLS + pooled grid at full width, before PCA (kept on the GPU host)."""
    base = config_dir(cfg) / "raw" / split
    return base / f"{archive}.cls.npy", base / f"{archive}.grid.npy"


def _atomic_memmap(path, shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    return tmp, np.lib.format.open_memmap(tmp, "w+", np.float16, shape)


def extract_raw_shard(cfg: FeatureConfig, split: str, archive: str, forward, dim: int,
                      manifest: Manifest, batch: int, workers: int, force: bool = False) -> dict | None:
    """Stage 1 (GPU): backbone over every frame of a shard; row i = row i of the frame index."""
    import torch

    kind, key = f"features_raw:{cfg.name}", f"{split}/{archive}"
    if manifest.state(kind, key) == "verified" and not force:
        return None
    tar, index = shard_paths(split, archive)
    idx = pd.read_parquet(index, columns=["f0_path", "offset", "size"])
    n = len(idx)
    cls_path, grid_path = raw_paths(cfg, split, archive)
    manifest.set(kind, key, "pending")
    tmp_cls, cls_out = _atomic_memmap(cls_path, (n, dim))
    tmp_grid, grid_out = _atomic_memmap(grid_path, (n, *cfg.grid, dim))
    t0, row = time.monotonic(), 0
    with torch.inference_mode():
        for imgs in _loader(list(zip([str(tar)] * n, idx["offset"], idx["size"])), cfg, batch, workers):
            cls, grid = forward(imgs)
            b = len(imgs)
            cls_out[row:row + b] = cls.half().cpu().numpy()
            grid_out[row:row + b] = grid.half().cpu().numpy()
            row += b
    seconds = time.monotonic() - t0
    cls_out.flush()
    grid_out.flush()
    del cls_out, grid_out
    problems = verify_features(tmp_cls, tmp_grid, n)
    if problems:
        manifest.set(kind, key, "failed", problems=problems)
        raise RuntimeError(f"raw features {key}: {problems}")
    tmp_cls.replace(cls_path)
    tmp_grid.replace(grid_path)
    stats = {"frames": n, "seconds": round(seconds, 1), "fps": round(n / max(seconds, 1e-9), 1)}
    manifest.set(kind, key, "verified", **stats)
    log.info("raw features %s: %d frames in %.0f s (%.0f img/s)", key, n, seconds, stats["fps"])
    return stats


def ensure_pca(cfg: FeatureConfig, manifest: Manifest, train_paths, device: str,
               strict: bool = True, splits=TASK_SPLITS) -> dict | None:
    """Load or fit the grid PCA from the stage-1 grids of a fixed frame subset.

    strict: every frame of the subset must have raw features, so the fit does not depend on
    which shards happen to be present; returns None (wait) otherwise. False only for tests."""
    out = config_dir(cfg) / "pca.npz"
    if out.exists():
        z = np.load(out)
        return {"mean": z["mean"], "components": z["components"], "explained": float(z["explained"])}
    import torch

    raw = []
    for key in manifest.keys(f"features_raw:{cfg.name}", "verified"):
        split, archive = key.split("/", 1)
        if split in splits:
            idx = pd.read_parquet(shard_paths(split, archive)[1], columns=["f0_path"])
            raw.append(idx.assign(split=split, archive=archive, row=np.arange(len(idx))))
    table = (pd.concat(raw, ignore_index=True).drop_duplicates("f0_path").set_index("f0_path")
             if raw else pd.DataFrame(columns=["split", "archive", "row"]))
    have = set(table.index)
    if not strict:
        train_paths = [p for p in train_paths if p in have]
    wanted = pca_subset(train_paths, cfg.pca_frames)
    missing = sum(p not in have for p in wanted)
    if missing or not wanted:
        log.info("PCA waiting: %d/%d subset frames have raw features", len(wanted) - missing, len(wanted))
        return None
    sel = table.loc[wanted]
    vecs = []
    for (split, archive), g in sel.groupby(["split", "archive"]):
        grid = np.load(raw_paths(cfg, split, archive)[1], mmap_mode="r")
        rows = np.sort(g["row"].to_numpy())
        vecs.append(torch.from_numpy(np.asarray(grid[rows], dtype=np.float32)).reshape(-1, grid.shape[-1]))
    pca = fit_pca(torch.cat(vecs).to(device), cfg.pca_dim)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("pca.partial.npz")
    np.savez(tmp, mean=pca["mean"], components=pca["components"], explained=pca["explained"])
    tmp.replace(out)
    log.info("PCA fitted on %d frames: %d dims keep %.1f%% of grid variance", len(sel),
             cfg.pca_dim, 100 * pca["explained"])
    return pca


def project_shard(cfg: FeatureConfig, split: str, archive: str, pca: dict, manifest: Manifest,
                  device: str, force: bool = False, chunk: int = 4096) -> dict | None:
    """Stage 2: final features = raw CLS + PCA-projected raw grid."""
    import torch

    kind, key = f"features:{cfg.name}", f"{split}/{archive}"
    if manifest.state(kind, key) == "verified" and not force:
        return None
    raw_info = manifest.get(f"features_raw:{cfg.name}", key)[1]
    raw_cls_path, raw_grid_path = raw_paths(cfg, split, archive)
    raw_cls, raw_grid = np.load(raw_cls_path, mmap_mode="r"), np.load(raw_grid_path, mmap_mode="r")
    n = len(raw_cls)
    cls_path, grid_path, index_path = feature_paths(cfg, split, archive)
    tmp_cls, cls_out = _atomic_memmap(cls_path, raw_cls.shape)
    tmp_grid, grid_out = _atomic_memmap(grid_path, (n, *cfg.grid, cfg.pca_dim))
    mean = torch.from_numpy(pca["mean"]).to(device)
    comps = torch.from_numpy(pca["components"]).to(device)
    for s in range(0, n, chunk):
        cls_out[s:s + chunk] = raw_cls[s:s + chunk]
        g = torch.from_numpy(np.asarray(raw_grid[s:s + chunk], dtype=np.float32)).to(device)
        grid_out[s:s + chunk] = ((g - mean) @ comps.T).half().cpu().numpy()
    cls_out.flush()
    grid_out.flush()
    del cls_out, grid_out
    problems = verify_features(tmp_cls, tmp_grid, n)
    if problems:
        manifest.set(kind, key, "failed", problems=problems)
        raise RuntimeError(f"features {key}: {problems}")
    tmp_cls.replace(cls_path)
    tmp_grid.replace(grid_path)
    idx = pd.read_parquet(shard_paths(split, archive)[1], columns=["f0_path"])
    tmp_idx = index_path.with_name(index_path.name + ".partial")
    pd.DataFrame({"f0_path": idx["f0_path"].to_numpy(), "row": np.arange(n)}).to_parquet(tmp_idx, index=False)
    tmp_idx.replace(index_path)
    stats = {"frames": n, "seconds": raw_info["seconds"], "fps": raw_info["fps"],
             "bytes": cls_path.stat().st_size + grid_path.stat().st_size}
    manifest.set(kind, key, "verified", **stats)
    log.info("features %s: %d frames, %.1f KB/frame", key, n, stats["bytes"] / max(n, 1) / 1e3)
    return stats


def verify_features(cls_path, grid_path, n: int) -> list[str]:
    problems = []
    for p in (cls_path, grid_path):
        a = np.load(p, mmap_mode="r")
        if a.shape[0] != n:
            problems.append(f"{p.name}: {a.shape[0]} rows, expected {n}")
        for s in range(0, a.shape[0], 4096):  # chunked: never load a whole shard
            if not np.isfinite(a[s:s + 4096]).all():
                problems.append(f"{p.name}: non-finite values")
                break
    return problems


def run(cfg: FeatureConfig, manifest: Manifest, device: str, batch: int = 64, workers: int = 8,
        shards: int | None = None, pretrained: bool = True, pca_train_table: str = "navtrain",
        pca_strict: bool = True, splits=TASK_SPLITS) -> list[dict]:
    """Stage 1 on every verified frame shard of `splits` (or the first `shards`), then, once
    the PCA can be fitted, stage 2 on every shard with raw features. Returns stage-2 stats."""
    from .openscene.samples import load_samples
    from .trajectory import HIST

    todo = frame_shards(manifest, splits)[:shards]
    forward = dim = None
    for split, archive in todo:
        if manifest.state(f"features_raw:{cfg.name}", f"{split}/{archive}") != "verified":
            if forward is None:
                forward, dim = build_backbone(cfg, device, pretrained=pretrained)
            extract_raw_shard(cfg, split, archive, forward, dim, manifest, batch, workers)
    smp = load_samples(pca_train_table)
    train_paths = smp.loc[smp["subset"] == "train", f"hist_f0_{HIST - 1}"]  # current frames
    pca = ensure_pca(cfg, manifest, train_paths, device, strict=pca_strict, splits=splits)
    if pca is None:
        return []
    write_meta(cfg, pca)
    out = []
    for split, archive in todo:
        if manifest.state(f"features_raw:{cfg.name}", f"{split}/{archive}") == "verified":
            stats = project_shard(cfg, split, archive, pca, manifest, device)
            if stats:
                out.append({"shard": f"{split}/{archive}", **stats})
    return out


def write_meta(cfg: FeatureConfig, pca: dict) -> None:
    import timm
    import torch

    meta = {**asdict(cfg), "feature_dim": int(pca["mean"].shape[0]), "pca_explained": pca["explained"],
            "timm": timm.__version__, "torch": torch.__version__}
    p = config_dir(cfg) / "meta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2))


def feature_index(cfg: FeatureConfig, manifest: Manifest) -> pd.DataFrame:
    """f0_path -> (split, archive, row) over all verified feature shards, first shard wins."""
    parts = []
    for key in manifest.keys(f"features:{cfg.name}", "verified"):
        split, archive = key.split("/", 1)
        idx = pd.read_parquet(feature_paths(cfg, split, archive)[2])
        parts.append(idx.assign(split=split, archive=archive))
    if not parts:
        return pd.DataFrame(columns=["f0_path", "row", "split", "archive"])
    return pd.concat(parts, ignore_index=True).drop_duplicates("f0_path", ignore_index=True)


def coverage(cfg: FeatureConfig, manifest: Manifest, tables: list[str], splits=TASK_SPLITS) -> dict:
    """For each sample table: history frames missing from the frame shards / the features."""
    from .openscene.samples import load_samples
    from .trajectory import HIST

    have_frames = set(frame_table(manifest, splits)["f0_path"])
    have_feats = set(feature_index(cfg, manifest)["f0_path"])
    out = {}
    for name in tables:
        smp = load_samples(name)
        need = set().union(*(set(smp[f"hist_f0_{k}"]) for k in range(HIST)))
        out[name] = {"frames_needed": len(need), "missing_frames": len(need - have_frames),
                     "missing_features": len(need - have_feats)}
    return out


class FeatureStore:
    """Random access to stored features by frame path, over memory-mapped shards."""

    KINDS = {"cls": 0, "grid": 1}

    def __init__(self, cfg: FeatureConfig, manifest: Manifest):
        self.cfg = cfg
        self.index = feature_index(cfg, manifest).set_index("f0_path")
        self._arrays: dict = {}

    def __contains__(self, f0_path: str) -> bool:
        return f0_path in self.index.index

    def _array(self, split: str, archive: str, kind: str) -> np.ndarray:
        k = (split, archive, kind)
        if k not in self._arrays:
            self._arrays[k] = np.load(feature_paths(self.cfg, split, archive)[self.KINDS[kind]], mmap_mode="r")
        return self._arrays[k]

    def get(self, f0_paths, kind: str = "cls") -> np.ndarray:
        """Features of `f0_paths` in that order (KeyError if a frame has none)."""
        r = self.index.loc[list(f0_paths)].reset_index(drop=True)
        out = None
        for (split, archive), g in r.groupby(["split", "archive"], sort=False):
            a = self._array(split, archive, kind)
            if out is None:
                out = np.empty((len(r),) + a.shape[1:], a.dtype)
            order = np.argsort(g["row"].to_numpy())  # sorted reads from the memmap
            out[g.index.to_numpy()[order]] = a[g["row"].to_numpy()[order]]
        return out
