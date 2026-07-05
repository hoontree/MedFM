"""Remove config keys that are empty/missing across ALL runs in a W&B project.

These show up as blank columns in the workspace and just add clutter. A key is
deleted only if it is empty (None/""/[]/{}) in every run that has it and never
holds a real value anywhere.

Usage:
    uv run tools/prune_empty_config.py --project distill --dry-run
    uv run tools/prune_empty_config.py --project distill
    uv run tools/prune_empty_config.py --project distill --keys student.r_d teacher.checkpoint
"""

import argparse
import time
from collections import defaultdict

import wandb

EMPTY = (None, "", [], {})


def flat(d, prefix=""):
    for k, v in d.items():
        nk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from flat(v, nk)
        else:
            yield nk, v


def with_retry(fn, tries=5, wait=5):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - transient 500/timeout from W&B
            if i == tries - 1:
                raise
            print(f"  ! {type(e).__name__}, retry {i + 1}/{tries} in {wait}s")
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="hheo")
    ap.add_argument("--project", default="distill")
    ap.add_argument("--keys", nargs="*", help="explicit keys to delete (skip auto-detect)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = wandb.Api(timeout=60)
    runs = with_retry(lambda: list(api.runs(f"{args.entity}/{args.project}")))
    print(f"runs: {len(runs)}")

    # Cache each run's flattened config once.
    run_cfgs = []
    for r in runs:
        cfg = with_retry(lambda r=r: dict(flat(dict(r.config))))
        run_cfgs.append((r, cfg))

    if args.keys:
        targets = set(args.keys)
    else:
        stats = defaultdict(lambda: {"empty": 0, "nonempty": 0})
        for _, cfg in run_cfgs:
            for k, v in cfg.items():
                stats[k]["empty" if v in EMPTY else "nonempty"] += 1
        targets = {k for k, c in stats.items() if c["nonempty"] == 0}
        print(f"\nAlways-empty keys to delete ({len(targets)}):")
        for k in sorted(targets):
            print(f"  {k}")

    if not targets:
        print("nothing to delete")
        return

    if args.dry_run:
        print("\n[dry-run] no changes written")
        return

    for r, cfg in run_cfgs:
        present = [k for k in targets if k in cfg]
        if not present:
            continue
        removed = 0
        for k in present:
            if delete_nested(r.config, k.split(".")):
                removed += 1
        if removed:
            with_retry(r.update)
            print(f"  updated {r.id}: removed {removed} keys")
    print("done")


def delete_nested(d, path):
    """Delete a nested key given its dotted path components. Returns True if removed."""
    cur = d
    for part in path[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    if isinstance(cur, dict) and path[-1] in cur:
        del cur[path[-1]]
        return True
    return False


if __name__ == "__main__":
    main()
