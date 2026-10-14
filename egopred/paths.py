"""Filesystem layout. Roots can be overridden with EGOPRED_DATA / EGOPRED_OUT."""

import os
from pathlib import Path

DATA = Path(os.environ.get("EGOPRED_DATA", "data"))
OUT = Path(os.environ.get("EGOPRED_OUT", "outputs"))

MANIFEST = DATA / "manifest.sqlite"
RAW = DATA / "raw"
OPENSCENE_META = DATA / "openscene" / "meta"  # one parquet per log: <split>/<log>.parquet
OPENSCENE_FRAMES = DATA / "openscene" / "frames"  # front-camera shards: <split>/<archive>.tar
SCENE_FILTERS = DATA / "openscene" / "scene_filters"
SAMPLES = DATA / "samples"  # <name>.parquet sample tables (targets, kinematics, tags)
FEATURES = DATA / "features"  # <config name>/<split>/<archive>.{cls,grid}.npy + index, pca.npz, meta.json
