"""Command line entry point: python -m egopred <command> ..."""

import argparse
import json
import logging

from . import paths
from .manifest import Manifest


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="egopred")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="manifest summary")
    mm = sub.add_parser("merge-manifest", help="import items from another manifest (newest row wins)")
    mm.add_argument("path")

    m = sub.add_parser("meta", help="stream OpenScene metadata of a split into per-log parquet")
    m.add_argument("--split", required=True, choices=["mini", "trainval", "test"])
    m.add_argument("--force", action="store_true")

    f = sub.add_parser("frames", help="stream an all-camera archive, keep CAM_F0 as a tar shard")
    f.add_argument("--split", required=True, choices=["mini", "trainval", "test", "navtrain"])
    f.add_argument("--index", type=int, required=True, help="archive number")
    f.add_argument("--kind", choices=["current", "history"], help="navtrain archive kind")
    f.add_argument("--max-width", type=int, default=None, help="downscale frames (default: keep bytes)")
    f.add_argument("--only-samples", help="keep only frames used by this sample table (e.g. navtest)")
    f.add_argument("--force", action="store_true")

    ad = sub.add_parser("frames-adopt", help="verify and register shards copied without their manifest")
    ad.add_argument("--split", required=True)

    s = sub.add_parser("samples", help="build a sample table from metadata (+ optional NAVSIM scene filter)")
    s.add_argument("--split", required=True, choices=["mini", "trainval", "test"])
    s.add_argument("--name", required=True)
    s.add_argument("--filter", help="NAVSIM scene filter name, e.g. navtrain, navtest, navmini")

    sub.add_parser("g0-comma", help="G0 on the comma2k19 example segment")
    g = sub.add_parser("g0-navsim", help="G0 on the logs of one ingested front-camera shard")
    g.add_argument("--split", default="mini")
    g.add_argument("--archive", default="openscene_sensor_mini_camera_0")

    fe = sub.add_parser("features", help="frozen-backbone features for every verified frame shard")
    fe.add_argument("--device", default="cuda")
    fe.add_argument("--batch", type=int, default=64)
    fe.add_argument("--workers", type=int, default=8, help="JPEG decoding processes")
    fe.add_argument("--shards", type=int, help="only the first N shards (throughput check)")
    fe.add_argument("--pca-table", default="navtrain", help="sample table whose train frames fit the PCA")
    fe.add_argument("--pca-partial", action="store_true", help="fit the PCA on whatever is ingested (smoke)")

    cv = sub.add_parser("coverage", help="history frames of sample tables missing from frames/features")
    cv.add_argument("names", nargs="+")

    g2 = sub.add_parser("g2", help="feature throughput/size + similarity sanity check")
    g2.add_argument("--table", default="navtest")

    b = sub.add_parser("g1", help="B0 baselines by scenario on sample tables")
    b.add_argument("names", nargs="+")

    def model_args(q, variant="C"):
        q.add_argument("--variant", default=variant, choices=["A", "B", "C"])
        q.add_argument("--vision", default="grid", choices=["grid", "cls"])
        q.add_argument("--k", type=int, default=6)
        q.add_argument("--d", type=int, default=256)
        q.add_argument("--layers", type=int, default=4)
        q.add_argument("--dropout", type=float, default=0.1)
        q.add_argument("--anchors", action="store_true", help="k-means anchor trajectories per mode")
        q.add_argument("--device", default="cuda")

    g3 = sub.add_parser("g3", help="gate G3: overfit ~1000 train samples (variant C)")
    model_args(g3)
    g3.add_argument("--n", type=int, default=1000)
    g3.add_argument("--epochs", type=int, default=200)
    g3.add_argument("--frames", default="fake", choices=["fake", "real"])
    g3.add_argument("--batch", type=int, default=32)
    g3.add_argument("--target-ade", type=float, default=0.5, help="training minADE to pass (m)")

    tr = sub.add_parser("train", help="train one variant on features, evaluate on val + test")
    model_args(tr)
    tr.add_argument("--name", required=True)
    tr.add_argument("--fraction", type=float, default=1.0, help="of the train samples (G4: 0.1)")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch", type=int, default=256)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight-decay", type=float, default=0.01)
    tr.add_argument("--max-minutes", type=float, default=0.0)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--test-table", default="navtest")
    tr.add_argument("--trivial-keep", type=float, default=1.0,
                    help="share of trivial train samples kept (B0-CTRV ADE < --trivial-ade)")
    tr.add_argument("--trivial-ade", type=float, default=0.5)
    tr.add_argument("--aug-noise", type=float, default=0.0, help="feature noise, in feature std units")
    tr.add_argument("--aug-frame-drop", type=float, default=0.0, help="drop prob. of each past frame")
    tr.add_argument("--aug-token-drop", type=float, default=0.0, help="drop prob. of each grid cell")
    tr.add_argument("--aug-vision-drop", type=float, default=0.0, help="drop prob. of all visual inputs")
    tr.add_argument("--future", default="", help="auxiliary loss: feature change of the frames at these s after t0, e.g. 1,2")
    tr.add_argument("--future-weight", type=float, default=0.1)
    tr.add_argument("--future-cond", action="store_true", help="condition the auxiliary head on the GT trajectory")

    ex = sub.add_parser("export", help="K trajectories + probabilities of runs' best checkpoints, val + test")
    ex.add_argument("names", nargs="+")
    ex.add_argument("--device", default="cuda")

    cp = sub.add_parser("compare", help="runs side by side by scenario, with B0")
    cp.add_argument("names", nargs="+")
    cp.add_argument("--split", default="val", choices=["val", "test"])

    fg = sub.add_parser("figure", help="README figure: frame + perspective view, two runs' predicted paths")
    fg.add_argument("--a", default="full_A_anch_lr1e4")
    fg.add_argument("--c", default="full_C_anch_lr1e4")
    fg.add_argument("--out", default="outputs/figures/compare.png")
    fg.add_argument("--sample", help="sample_id (default: picked by --scenario/--pick)")
    fg.add_argument("--scenario")
    fg.add_argument("--pick", default="median", choices=["gain", "loss", "median", "random"])
    fg.add_argument("--seed", type=int, default=0)

    gf = sub.add_parser("gif", help="animated figure: consecutive 2 Hz samples of one log, two runs' paths")
    gf.add_argument("--a", default="full_A_anch_lr1e4")
    gf.add_argument("--c", default="full_C_anch_lr1e4")
    gf.add_argument("--out", default="outputs/figures/compare.gif")
    gf.add_argument("--length", type=int, default=24, help="number of consecutive samples (0.5 s each)")
    gf.add_argument("--pick", default="gain", choices=["gain", "loss", "random"])
    gf.add_argument("--seed", type=int, default=0)
    gf.add_argument("--hz", type=float, default=2.0)
    gf.add_argument("--scenario", help="only windows mostly of this scenario tag")

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    manifest = Manifest(paths.MANIFEST)

    if a.cmd == "status":
        print(json.dumps(manifest.summary(), indent=2))
    elif a.cmd == "merge-manifest":
        from pathlib import Path

        print(f"{manifest.merge(Path(a.path))} items imported")
    elif a.cmd == "meta":
        from .openscene.metadata import ingest_metadata

        ingest_metadata(a.split, manifest, force=a.force)
    elif a.cmd == "frames":
        from .openscene import sources
        from .openscene.frames import ingest_archive

        if a.split == "navtrain":
            if not a.kind:
                p.error("--kind is required for navtrain")
            archive, url = sources.navtrain_archive(a.kind, a.index)
            split = "trainval"
        else:
            archive, url = sources.camera_archive(a.split, a.index)
            split = a.split
        wanted = None
        if a.only_samples:
            from .openscene.samples import load_samples
            from .trajectory import HIST

            smp = load_samples(a.only_samples, columns=[f"hist_f0_{k}" for k in range(HIST)])
            wanted = set().union(*(set(smp[f"hist_f0_{k}"]) for k in range(HIST)))
        ingest_archive(split, archive, url, manifest, wanted=wanted, max_width=a.max_width, force=a.force)
    elif a.cmd == "frames-adopt":
        from .openscene.frames import adopt_shards

        print(json.dumps(adopt_shards(a.split, manifest), indent=1))
    elif a.cmd == "samples":
        from .openscene.samples import build_samples, scene_filter_tokens

        build_samples(a.split, a.name, scene_filter_tokens(a.filter) if a.filter else None)
    elif a.cmd == "g0-comma":
        from .g0 import run_comma

        print(json.dumps(run_comma(), indent=2))
    elif a.cmd == "g0-navsim":
        from .g0 import run_navsim

        print(json.dumps(run_navsim(a.split, a.archive), indent=2))
    elif a.cmd == "features":
        from .features import DEFAULT, run

        stats = run(DEFAULT, manifest, a.device, batch=a.batch, workers=a.workers, shards=a.shards,
                    pca_train_table=a.pca_table, pca_strict=not a.pca_partial)
        print(json.dumps(stats, indent=2))
    elif a.cmd == "coverage":
        from .features import DEFAULT, coverage

        print(json.dumps(coverage(DEFAULT, manifest, a.names), indent=2))
    elif a.cmd == "g2":
        from .g2 import report as g2_report

        print(json.dumps(g2_report(a.table), indent=2))
    elif a.cmd == "g1":
        from .g1 import report

        report(a.names)
    elif a.cmd in ("g3", "train"):
        from .model import ModelConfig
        from .train import Augment, Frames, TrainConfig, g3, hist_paths, run

        mcfg = ModelConfig(variant=a.variant, vision=a.vision, k=a.k, d=a.d, layers=a.layers,
                           dropout=a.dropout if a.cmd == "train" else 0.0, anchors=a.anchors)
        if a.cmd == "train" and a.future:
            mcfg.future = tuple(float(x) for x in a.future.split(","))
            mcfg.future_weight, mcfg.future_cond = a.future_weight, a.future_cond
        if a.cmd == "g3":
            print(json.dumps(g3(a.n, a.epochs, a.device, mcfg, a.frames, a.target_ade, a.batch), indent=2))
        else:
            from .features import DEFAULT, FeatureStore

            cfg = TrainConfig(name=a.name, model=mcfg, fraction=a.fraction, epochs=a.epochs, batch=a.batch,
                              lr=a.lr, weight_decay=a.weight_decay, max_minutes=a.max_minutes, seed=a.seed,
                              test_table=a.test_table, trivial_keep=a.trivial_keep, trivial_ade=a.trivial_ade,
                              aug=Augment(a.aug_noise, a.aug_frame_drop, a.aug_token_drop, a.aug_vision_drop))
            store = FeatureStore(DEFAULT, manifest)
            res = run(cfg, lambda tables: Frames.from_store(store, hist_paths(*tables)), a.device)
            print(json.dumps({k: res[k] for k in ("parameters", "train_samples")}, indent=2))
    elif a.cmd == "export":
        from .features import DEFAULT, FeatureStore
        from .train import Frames, export_predictions, hist_paths

        store = FeatureStore(DEFAULT, manifest)
        for name in a.names:
            n = export_predictions(name, lambda tables: Frames.from_store(store, hist_paths(*tables)), a.device)
            print(name, json.dumps(n))
    elif a.cmd == "compare":
        from .train import compare

        print(compare(a.names, a.split))
    elif a.cmd == "gif":
        from pathlib import Path

        from .figures import make_gif

        print(make_gif(a.a, a.c, Path(a.out), length=a.length, pick=a.pick, seed=a.seed, hz=a.hz, scenario=a.scenario))
    elif a.cmd == "figure":
        from pathlib import Path

        from .figures import make_figure

        print(make_figure(a.a, a.c, Path(a.out), sample=a.sample, scenario=a.scenario, pick=a.pick, seed=a.seed))
    manifest.close()


if __name__ == "__main__":
    main()
