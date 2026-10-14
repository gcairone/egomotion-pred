"""Gate G1: B0 baselines (constant velocity, constant turn rate) on the sample tables,
stratified by scenario, plus dataset statistics used for curation."""

import json
import logging

import numpy as np
import pandas as pd

from . import paths
from .openscene.samples import future, load_samples
from .trajectory import SCENARIOS, constant_turn_rate, constant_velocity, displacement_metrics

log = logging.getLogger(__name__)

BASELINES = {
    "cv": lambda s: constant_velocity(s[["vx", "vy"]].to_numpy()),
    "ctrv": lambda s: constant_turn_rate(s["speed"].to_numpy(), s["yaw_rate"].to_numpy()),
}
METRICS = ["ade", "fde", "miss", "lat"]


def evaluate(s: pd.DataFrame) -> pd.DataFrame:
    """Per-sample metrics of every baseline, as columns <baseline>_<metric>."""
    gt = future(s)
    out = pd.DataFrame(index=s.index)
    for name, fn in BASELINES.items():
        for k, v in displacement_metrics(fn(s), gt).items():
            if k in METRICS:
                out[f"{name}_{k}"] = v
    return out


def stratified(s: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([s[["scenario"]], m], axis=1)
    rows = []
    for scen in SCENARIOS + ["ALL", "ALL (moving)"]:
        if scen == "ALL":
            part = df
        elif scen == "ALL (moving)":
            part = df[df["scenario"] != "static"]
        else:
            part = df[df["scenario"] == scen]
        if len(part) == 0:
            continue
        row = {"scenario": scen, "n": len(part), "share_%": 100 * len(part) / len(df)}
        for c in m.columns:
            row[c] = part[c].mean() * (100 if c.endswith("miss") else 1)
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_stats(s: pd.DataFrame) -> dict:
    q = [5, 25, 50, 75, 95]
    return {
        "samples": len(s),
        "logs": int(s["log_name"].nunique()),
        "drives": int(s["drive_id"].nunique()) if "drive_id" in s else None,
        "speed_pctl": dict(zip(q, np.percentile(s["speed"], q).round(2).tolist())),
        "abs_heading_change_deg_pctl": dict(zip(q, np.percentile(np.abs(s["heading_change_deg"]), q).round(2).tolist())),
        "gap_samples": int(s["gap"].sum()),
        "pose_jump_samples": int(s["pose_jump"].sum()),
        "command_counts": {int(k): int(v) for k, v in s["command"].value_counts().sort_index().items()},
        "city_counts": {str(k): int(v) for k, v in s["map_location"].value_counts().items()},
        "scenario_flags_%": {k: round(100 * float(s[f"is_{k}"].mean()), 2) for k in SCENARIOS},
    }


def md_table(df: pd.DataFrame) -> str:
    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def report(names: list[str]) -> dict:
    out_dir = paths.OUT / "g1"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# G1: B0 baselines by scenario", "",
          "Errors in metres over the 4 s horizon (8 waypoints at 2 Hz); miss = FDE > 2 m (%).",
          "Baselines: cv = constant ego-frame velocity; ctrv = constant speed and yaw rate.", ""]
    result = {}
    for name in names:
        s = load_samples(name)
        evaluated = {name: s}
        if "subset" in s and name != "navtest":
            evaluated = {f"{name}/{k}": s[s["subset"] == k] for k in ("train", "val")}
        for label, part in evaluated.items():
            part = part.reset_index(drop=True)
            m = evaluate(part)
            table = stratified(part, m)
            stats = dataset_stats(part)
            result[label] = {"stats": stats, "by_scenario": table.to_dict(orient="records")}
            md += [f"## {label}", "", f"{stats['samples']} samples, {stats['logs']} logs, "
                   f"{stats['drives']} drives; gaps {stats['gap_samples']}, pose jumps {stats['pose_jump_samples']}.",
                   "", md_table(table), ""]
            log.info("%s\n%s", label, table.round(2).to_string(index=False))
    (out_dir / "b0_report.md").write_text("\n".join(md))
    (out_dir / "b0_report.json").write_text(json.dumps(result, indent=2, default=float))
    return result
