"""Gate G2: feature extraction measured (throughput, size on disk) and sanity-checked.

Sanity check: features of frames 0.5 s apart in the same log must be much more similar than
features of the current frames of two samples from different logs; otherwise the features
(or the frame <-> path association) are broken.
"""

import json
import logging

import numpy as np

from . import paths
from .features import DEFAULT, FeatureStore, config_dir
from .manifest import Manifest
from .openscene.samples import load_samples
from .trajectory import HIST

log = logging.getLogger(__name__)


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.reshape(len(a), -1).astype(np.float32)
    b = b.reshape(len(b), -1).astype(np.float32)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def report(table: str = "navtest", n: int = 2000, cfg=DEFAULT) -> dict:
    manifest = Manifest(paths.MANIFEST)
    kind = f"features:{cfg.name}"
    shards = [manifest.get(kind, k)[1] for k in manifest.keys(kind, "verified")]
    frames = sum(s["frames"] for s in shards)
    nbytes = sum(s["bytes"] for s in shards)
    seconds = sum(s["seconds"] for s in shards)
    meta = json.loads((config_dir(cfg) / "meta.json").read_text())

    store = FeatureStore(cfg, manifest)
    smp = load_samples(table)
    smp = smp.sample(min(n, len(smp)), random_state=0).reset_index(drop=True)
    cur, prev = smp[f"hist_f0_{HIST - 1}"], smp[f"hist_f0_{HIST - 2}"]
    other = smp.sample(frac=1.0, random_state=1).reset_index(drop=True)
    diff_log = (other["log_name"] != smp["log_name"]).to_numpy()
    checks = {}
    for k in ("cls", "grid"):
        a, b, c = store.get(cur, k), store.get(prev, k), store.get(other[f"hist_f0_{HIST - 1}"], k)
        checks[k] = {"cos_0.5s_same_log": float(_cos(a, b).mean()),
                     "cos_other_log": float(_cos(a[diff_log], c[diff_log]).mean())}
    out = {
        "config": meta,
        "shards": len(shards),
        "frames": frames,
        "gb_on_disk": round(nbytes / 1e9, 2),
        "kb_per_frame": round(nbytes / max(frames, 1) / 1e3, 1),
        "gpu_img_per_s": round(frames / max(seconds, 1e-9), 1),
        "extraction_minutes": round(seconds / 60, 1),
        "pca_explained_variance": round(meta["pca_explained"], 4),
        "sanity": checks,
        "sanity_ok": all(v["cos_0.5s_same_log"] > v["cos_other_log"] + 0.1 for v in checks.values()),
    }
    dest = paths.OUT / "g2"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "report.json").write_text(json.dumps(out, indent=2))
    manifest.close()
    return out
