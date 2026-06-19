"""Aggregate, analyse and visualise the meta-reliability sweep.

Builds on ``tools/summarize_reliability_sweep.py`` (reuses its metric loader)
and adds, for the ``config/sweeps/meta_reliability.yaml`` sweep:

  * a method-comparison bar chart (external + internal mean Dice), with the
    core methods (`base_*`, `handcrafted_reliability`, `learned_pseudo`,
    `meta_scalar`, `meta_mixture`) highlighted;
  * a per-dataset grouped-bar breakdown for those core methods;
  * reliability/meta trajectories parsed from each run's epoch logs
    (`mean_reliability`, `meta/meta_sup_loss`) — also the H4 collapse evidence;
  * an `analysis.md` with the H1 / H2 / H4 verdicts.

Runs incrementally: only completed runs (those with parsed metrics) are plotted,
so it is safe to call repeatedly while the sweep is still in flight.

Usage:
    uv run tools/visualize_meta_reliability.py                      # newest sweep
    uv run tools/visualize_meta_reliability.py logs/reliability_ablation/<ts>
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from tools.summarize_reliability_sweep import (  # noqa: E402
    EXTERNAL,
    INTERNAL,
    _mean,
    load_metrics,
)

# Canonical display order / grouping for the meta sweep.
CORE_ORDER = [
    "base_task_only",
    "base_logit_kd",
    "handcrafted_reliability",
    "learned_pseudo",
    "meta_scalar",
    "meta_mixture",
]
ABLATION_ORDER = [
    "meta_full_features",
    "meta_split_val",
    "meta_no_prior",
    "meta_no_sparsity",
    "meta_no_reg",
    "meta_rho_0.3",
    "meta_rho_0.7",
    "meta_staged",
]
# Colour key: baselines grey, hand-crafted orange, learned blue, meta green/red.
COLORS = {
    "base_task_only": "#9e9e9e",
    "base_logit_kd": "#bdbdbd",
    "handcrafted_reliability": "#ff9800",
    "learned_pseudo": "#2196f3",
    "meta_scalar": "#2e7d32",
    "meta_mixture": "#66bb6a",
}
DEFAULT_COLOR = "#c5cae9"

_EPOCH_TRAIN = re.compile(r"mean_reliability: ([\d.]+)")
_EPOCH_METASUP = re.compile(r"meta/meta_sup_loss: ([\d.]+)")


def _ext(per_ds, metric):
    return _mean([per_ds[d][metric] for d in EXTERNAL if d in per_ds and metric in per_ds[d]])


def _int(per_ds, metric):
    return _mean([per_ds[d][metric] for d in INTERNAL if d in per_ds and metric in per_ds[d]])


def parse_trajectory(log_path: Path):
    """Per-epoch (mean_reliability, meta_sup_loss) lists from a run's epoch log.

    Reads the dispatcher-captured ``<name>.log`` (one ``Train: ...`` line per
    epoch). Either series may be empty (non-meta runs have no meta_sup_loss).
    """
    rel, meta = [], []
    if not log_path.exists():
        return rel, meta
    for line in log_path.read_text(errors="ignore").splitlines():
        m = _EPOCH_TRAIN.search(line)
        if m:
            rel.append(float(m.group(1)))
        m2 = _EPOCH_METASUP.search(line)
        if m2:
            meta.append(float(m2.group(1)))
    return rel, meta


def find_sweep_dir(argv):
    if len(argv) > 1:
        return Path(argv[1])
    cands = sorted((PROJECT_DIR / "logs" / "reliability_ablation").glob("*/"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    cands = [c for c in cands if list(c.glob("*.log"))]
    if not cands:
        sys.exit("No sweep dir with logs found.")
    return cands[0]


def plot_method_comparison(metrics, sweep_dir):
    """Bar chart of external + internal mean Dice per method (completed runs)."""
    order = [n for n in CORE_ORDER + ABLATION_ORDER if n in metrics] + \
            [n for n in metrics if n not in CORE_ORDER + ABLATION_ORDER]
    names = [n for n in order if _ext(metrics[n], "Dice") is not None]
    if not names:
        return None
    ext = [_ext(metrics[n], "Dice") for n in names]
    intn = [_int(metrics[n], "Dice") for n in names]

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(names) + 3), 5))
    x = range(len(names))
    colors = [COLORS.get(n, DEFAULT_COLOR) for n in names]
    ax.bar([i - 0.2 for i in x], ext, width=0.4, color=colors, label="external mean")
    ax.bar([i + 0.2 for i in x], intn, width=0.4, color=colors, alpha=0.45,
           hatch="//", label="internal mean")
    for i, v in enumerate(ext):
        if v is not None:
            ax.text(i - 0.2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Dice")
    ax.set_title(f"Meta-Reliability sweep — test Dice by method ({sweep_dir.name})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = sweep_dir / "fig_method_comparison.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_per_dataset(metrics, sweep_dir):
    """Grouped bars: per-dataset Dice for the core methods."""
    names = [n for n in CORE_ORDER if n in metrics]
    datasets = [d for d in INTERNAL + EXTERNAL
                if any(d in metrics[n] for n in names)]
    if not names or not datasets:
        return None
    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(datasets) + 2), 5))
    w = 0.8 / len(names)
    for j, n in enumerate(names):
        vals = [metrics[n].get(d, {}).get("Dice") for d in datasets]
        xs = [i + j * w - 0.4 + w / 2 for i in range(len(datasets))]
        ax.bar(xs, [v if v is not None else 0 for v in vals], width=w,
               label=n, color=COLORS.get(n, DEFAULT_COLOR))
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
    ax.axvline(len(INTERNAL) - 0.5, color="k", ls=":", alpha=0.5)
    ax.set_ylabel("Dice")
    ax.set_title("Per-dataset Dice (internal | external)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = sweep_dir / "fig_per_dataset.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_trajectories(sweep_dir):
    """Reliability + meta-loss trajectories for every meta/learned run."""
    runs = {}
    for lp in sorted(sweep_dir.glob("*.log")):
        name = lp.stem
        if name == "index":
            continue
        rel, meta = parse_trajectory(lp)
        if rel:
            runs[name] = (rel, meta)
    runs = {n: v for n, v in runs.items()
            if n.startswith("meta") or n == "learned_pseudo" or n == "handcrafted_reliability"}
    if not runs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, (rel, meta) in sorted(runs.items()):
        c = COLORS.get(name, DEFAULT_COLOR)
        axes[0].plot(range(1, len(rel) + 1), rel, label=name, color=c, alpha=0.85)
        if meta:
            axes[1].plot(range(1, len(meta) + 1), meta, label=name, color=c, alpha=0.85)
    axes[0].axhline(0.05, color="r", ls=":", alpha=0.6, label="collapse (0.05)")
    axes[0].set_title("mean reliability per epoch (H4: watch for collapse)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("mean r"); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].set_title("meta supervised loss per epoch")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("L_meta_sup"); axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    out = sweep_dir / "fig_trajectories.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _verdict(name, a, b, label_a, label_b, higher_better=True):
    if a is None or b is None:
        missing = label_a if a is None else label_b
        return f"- **{name}: pending** ({missing} not finished)."
    gap = a - b
    win = (gap > 0) if higher_better else (gap < 0)
    mark = "✅" if win else "❌"
    return (f"- **{name}: {mark}** {label_a}={a:.4f} vs {label_b}={b:.4f} "
            f"(Δ={gap:+.4f}).")


def write_analysis(metrics, sweep_dir):
    lines = [f"# Meta-Reliability sweep analysis — {sweep_dir.name}", ""]
    done = [n for n in metrics if _ext(metrics[n], "Dice") is not None]
    lines.append(f"{len(done)} runs with final metrics: {', '.join(sorted(done)) or '(none yet)'}")
    lines.append("")

    def ed(n):
        return _ext(metrics[n], "Dice") if n in metrics else None

    lines.append("## Hypothesis verdicts (external-mean Dice)")
    lines.append(_verdict("H1 learnable ≥ hand-crafted",
                          ed("meta_scalar"), ed("handcrafted_reliability"),
                          "meta_scalar", "handcrafted_reliability"))
    lines.append(_verdict("H2 meta > pseudo-label (central claim)",
                          ed("meta_scalar"), ed("learned_pseudo"),
                          "meta_scalar", "learned_pseudo"))
    lines.append(_verdict("meta_scalar > plain logit-KD",
                          ed("meta_scalar"), ed("base_logit_kd"),
                          "meta_scalar", "base_logit_kd"))

    # H4: collapse evidence from trajectories.
    rel_noreg, _ = parse_trajectory(sweep_dir / "meta_no_reg.log")
    lines.append("")
    lines.append("## H4 collapse check (regularisers)")
    if rel_noreg:
        last = rel_noreg[-1]
        coll = "✅ collapsed" if last < 0.05 else ("partial" if last < 0.2 else "❌ no collapse")
        lines.append(f"- meta_no_reg final mean r = {last:.4f} → {coll}.")
        if ed("meta_no_reg") is not None and ed("base_task_only") is not None:
            lines.append(_verdict("  meta_no_reg regresses toward task-only",
                                  ed("base_task_only"), ed("meta_no_reg"),
                                  "base_task_only", "meta_no_reg"))
    else:
        lines.append("- meta_no_reg: pending (no epoch logs yet).")

    # Full ranking table.
    lines += ["", "## Ranking (external-mean Dice)", "",
              "| rank | method | ext Dice | int Dice |", "|---:|---|---:|---:|"]
    rank = sorted(done, key=lambda n: ed(n) or -1, reverse=True)
    for i, n in enumerate(rank, 1):
        lines.append(f"| {i} | {n} | {ed(n):.4f} | "
                     f"{(_int(metrics[n], 'Dice') or float('nan')):.4f} |")

    out = sweep_dir / "analysis.md"
    out.write_text("\n".join(lines) + "\n")
    return out, "\n".join(lines)


def main():
    sweep_dir = find_sweep_dir(sys.argv)
    metrics = {name: per_ds for name, per_ds in load_metrics(sweep_dir) if per_ds}

    figs = [
        plot_method_comparison(metrics, sweep_dir),
        plot_per_dataset(metrics, sweep_dir),
        plot_trajectories(sweep_dir),
    ]
    analysis_path, analysis_text = write_analysis(metrics, sweep_dir)

    print(analysis_text)
    print("\nSaved:")
    print(f"  {analysis_path}")
    for f in figs:
        if f:
            print(f"  {f}")


if __name__ == "__main__":
    main()
