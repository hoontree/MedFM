"""Backfill final_test/{internal,external}_mean/* into existing W&B runs.

Reads each run's summary, finds final_test/{ds_name}/{metric} entries, groups
datasets into internal (names ending in `_test`) vs external, computes the mean
per metric, and writes them back as final_test/{group}_mean/{metric}.

Usage:
    uv run tools/backfill_group_mean.py                 # all runs in project
    uv run tools/backfill_group_mean.py --runs ID1 ID2  # specific run ids
    uv run tools/backfill_group_mean.py --dry-run
"""

import argparse
import re
from collections import defaultdict

import numpy as np
import wandb

PREFIX = "final_test/"
# matches: final_test/<ds_name>/<metric>
KEY_RE = re.compile(r"^final_test/(?P<ds>[^/]+)/(?P<metric>[^/]+)$")
# already-aggregated groups to skip when parsing per-dataset values
GROUP_NAMES = {"internal_mean", "external_mean"}


def compute_group_means(summary: dict) -> dict:
    # group -> metric -> {"mean": [...], "std": [...]} per dataset
    groups = {"internal": defaultdict(lambda: {"mean": [], "std": []}),
              "external": defaultdict(lambda: {"mean": [], "std": []})}
    for key, value in summary.items():
        m = KEY_RE.match(key)
        if not m:
            continue
        ds = m.group("ds")
        if ds in GROUP_NAMES:
            continue
        if not isinstance(value, (int, float)):
            continue
        metric = m.group("metric")
        group = "internal" if ds.endswith("_test") else "external"
        if metric.endswith("_std"):
            groups[group][metric[:-4]]["std"].append(value)
        else:
            groups[group][metric]["mean"].append(value)

    out = {}
    for group, per_metric in groups.items():
        for metric, vals in per_metric.items():
            means = np.array(vals["mean"], dtype=float)
            if means.size == 0:
                continue
            out[f"{PREFIX}{group}_mean/{metric}"] = float(means.mean())
            # Overall std via law of total variance (datasets weighted equally):
            #   std = sqrt(mean(std_i^2) + var(mean_i))
            stds = np.array(vals["std"], dtype=float)
            if stds.size == means.size:
                total_var = float((stds ** 2).mean() + means.var())
                out[f"{PREFIX}{group}_mean/{metric}_std"] = float(np.sqrt(total_var))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="hheo")
    ap.add_argument("--project", default="TinyUSFM")
    ap.add_argument("--runs", nargs="*", help="specific run ids (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = wandb.Api()
    if args.runs:
        runs = [api.run(f"{args.entity}/{args.project}/{rid}") for rid in args.runs]
    else:
        runs = api.runs(f"{args.entity}/{args.project}")

    for run in runs:
        summary = dict(run.summary)
        means = compute_group_means(summary)
        if not means:
            print(f"[skip] {run.id} ({run.name}): no final_test/* per-dataset metrics")
            continue
        pretty = ", ".join(f"{k.split('/',1)[1]}={v:.4f}" for k, v in means.items())
        print(f"[{'dry' if args.dry_run else 'write'}] {run.id} ({run.name}): {pretty}")
        if not args.dry_run:
            run.summary.update(means)
            run.summary.update()  # flush


if __name__ == "__main__":
    main()
