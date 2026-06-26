"""Aggregate a reliability-KD sweep into a comparison table.

Parses the per-experiment dispatcher logs written by
``tools/run_reliability_ablation.py`` (``<sweep_dir>/<name>.log``), extracts the
final-test metrics each run prints per dataset, and emits a sorted Markdown +
CSV table. Internal (BUSBRA/BUSI/B) and external (BUID/BUS_UCLM/...) datasets
are averaged separately; ranking is by external-mean Dice (the generalization
signal).

Usage:
    uv run tools/summarize_reliability_sweep.py                       # newest sweep
    uv run tools/summarize_reliability_sweep.py logs/reliability_ablation/<ts>
"""

import csv
import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
INTERNAL = ["BUSBRA_test", "BUSI_test", "B_test"]
EXTERNAL = ["BUID", "BUS_UCLM", "BUS_UCLM_filtered"]

_LINE = re.compile(
    r"final_test_(?P<ds>\S+) Metrics - Dice: (?P<dice>[\d.]+|nan), "
    r"HD95: (?P<hd95>[\d.]+|nan|inf), BFScore: (?P<bf>[\d.]+|nan), "
    r"ECE: [\d.]+, IOU: (?P<iou>[\d.]+|nan), "
    r"Sensitivity: (?P<sens>[\d.]+|nan)"
)

# experiment-name prefix -> ablation axis (mirrors run_reliability_ablation.py)
_AXIS = {
    "base_": "baseline", "b0_": "factor", "b1_": "factor", "b2_": "factor",
    "b3_": "factor", "lo_": "leave_one_out", "tg_": "gate_weight",
    "sb_": "gate_weight", "temp_": "temperature", "bs_": "batch_size",
    "task_": "main_method", "logit_": "main_method", "reliability": "main_method",
}


def _axis(name: str) -> str:
    for p, a in _AXIS.items():
        if name.startswith(p):
            return a
    return "misc"


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def parse_log(path: Path):
    """Return {dataset: {Dice,IoU,HD95,BFScore}} for the last final_test block."""
    per_ds = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = _LINE.search(line)
        if not m:
            continue
        f = lambda s: float(s) if s not in ("nan", "inf") else float("nan")
        per_ds[m["ds"]] = {
            "Dice": f(m["dice"]), "IoU": f(m["iou"]),
            "HD95": f(m["hd95"]), "BFScore": f(m["bf"]),
            "Sensitivity": f(m["sens"]),
        }
    return per_ds


def load_metrics(sweep_dir: Path):
    """Yield (name, per_ds) preferring final_metrics.json via index.jsonl.

    Robust path: the dispatcher's ``index.jsonl`` maps each run to its
    ``exp_dir``; we read the structured ``final_metrics.json`` there. Falls back
    to regex-parsing the per-run ``*.log`` (older runs without JSON).
    """
    index = sweep_dir / "index.jsonl"
    seen = set()
    if index.exists():
        for line in index.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            name = rec.get("experiment") or rec.get("run_name", "?")
            exp_dir = rec.get("exp_dir")
            jpath = Path(exp_dir) / "final_metrics.json" if exp_dir else None
            if jpath and jpath.exists():
                data = json.loads(jpath.read_text())
                per_ds = {k: v for k, v in data.get("metrics", {}).items()}
                seen.add(name)
                yield name, per_ds
    # Fallback / older sweeps: parse logs not already covered by the index.
    for lp in sorted(sweep_dir.glob("*.log")):
        if lp.stem in seen:
            continue
        yield lp.stem, parse_log(lp)


def main():
    if len(sys.argv) > 1:
        sweep_dir = Path(sys.argv[1])
    else:
        cands = sorted((PROJECT_DIR / "logs" / "reliability_ablation").glob("*/"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        cands = [c for c in cands if list(c.glob("*.log"))]
        if not cands:
            sys.exit("No sweep dir with logs found.")
        sweep_dir = cands[0]

    def ext(per_ds, metric):
        return _mean([per_ds[d][metric] for d in EXTERNAL if d in per_ds and metric in per_ds[d]])

    def intn(per_ds, metric):
        return _mean([per_ds[d][metric] for d in INTERNAL if d in per_ds and metric in per_ds[d]])

    rows = []
    for name, per_ds in load_metrics(sweep_dir):
        if not per_ds:
            rows.append({"name": name, "axis": _axis(name), "status": "no metrics"})
            continue
        rows.append({
            "name": name, "axis": _axis(name), "status": "ok",
            "ext_Dice": ext(per_ds, "Dice"), "int_Dice": intn(per_ds, "Dice"),
            "ext_Sens": ext(per_ds, "Sensitivity"),
            "ext_IoU": ext(per_ds, "IoU"), "ext_HD95": ext(per_ds, "HD95"),
            **{f"{d}_Dice": per_ds[d]["Dice"] for d in INTERNAL + EXTERNAL
               if d in per_ds and "Dice" in per_ds[d]},
        })

    if not rows:
        sys.exit(f"No runs found in {sweep_dir}")

    ok = [r for r in rows if r["status"] == "ok"]
    ok.sort(key=lambda r: (r.get("ext_Dice") or -1), reverse=True)

    # Markdown
    md = [f"# Reliability sweep summary — {sweep_dir.name}", "",
          f"{len(ok)}/{len(rows)} runs with metrics. Ranked by external-mean Dice.", "",
          "| rank | experiment | axis | ext Dice | int Dice | ext Sens | ext IoU | ext HD95 |",
          "|---:|---|---|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(ok, 1):
        md.append(f"| {i} | {r['name']} | {r['axis']} | {_fmt(r['ext_Dice'])} | "
                  f"{_fmt(r['int_Dice'])} | {_fmt(r['ext_Sens'])} | {_fmt(r['ext_IoU'])} | "
                  f"{_fmt(r['ext_HD95'])} |")
    missing = [r for r in rows if r["status"] != "ok"]
    if missing:
        md += ["", "Runs without parsed metrics: " + ", ".join(r["name"] for r in missing)]
    md_text = "\n".join(md)

    out_md = sweep_dir / "summary_table.md"
    out_md.write_text(md_text + "\n")

    # CSV (full per-dataset detail)
    cols = ["name", "axis", "status", "ext_Dice", "int_Dice", "ext_Sens",
            "ext_IoU", "ext_HD95", *[f"{d}_Dice" for d in INTERNAL + EXTERNAL]]
    with open(sweep_dir / "summary_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in ok + missing:
            w.writerow({k: (_fmt(r[k]) if isinstance(r.get(k), float) else r.get(k, "")) for k in cols})

    print(md_text)
    print(f"\nSaved: {out_md}\n       {sweep_dir / 'summary_table.csv'}")


if __name__ == "__main__":
    main()
