"""Aggregate controlled fine-tuning baselines into a SAM/TinyUSFM x sampling table.

Reads each train.py run directory's ``config.yaml`` (for model + sampling flag)
and ``test_results.txt`` (for external/internal metrics), then prints a 2x2-style
comparison and writes ``docs/kd_baseline_finetune.tex``.

Auto-detects the latest run per (model, sampling) cell under logs/train/ when no
dirs are given; or pass explicit run dirs:

    uv run tools/summarize_baselines.py
    uv run tools/summarize_baselines.py logs/train/sam/e_ft_d_ft/<ts>_... logs/train/TinyUSFM/<ts>_...
"""

import re
import statistics as st
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EXT = ["BUID", "BUS_UCLM", "BUS_UCLM_filtered"]
INT = ["BUSBRA_test", "BUSI_test", "B_test"]


def parse_test_results(p: Path):
    t = {}
    for ln in p.read_text(errors="ignore").splitlines():
        m = re.match(r"(\w+)/(Dice|Sensitivity|IoU):\s*([0-9.]+)$", ln.strip())
        if m:
            t.setdefault(m.group(1), {})[m.group(2)] = float(m.group(3))
    return t


def run_meta(run_dir: Path):
    """Return (model_label, sampling_bool) from config.yaml, or (None, None)."""
    cfg = run_dir / "config.yaml"
    if not cfg.exists():
        return None, None
    txt = cfg.read_text(errors="ignore")
    sampling = bool(re.search(r"enabled:\s*true", txt.split("sampling:")[-1][:120])) \
        if "sampling:" in txt else False
    name = None
    mname = re.search(r"name:\s*(sam\w*|TinyUSFM|tinyusfm\w*)", txt)
    enc = re.search(r"encoder_mode:\s*(\w+)", txt)
    if mname:
        nm = mname.group(1).lower()
        if "sam" in nm:
            name = f"SAM-{(enc.group(1) if enc else 'ft')}"
        else:
            name = "TinyUSFM"
    return name, sampling


def emean(t, keys, metric):
    vals = [t[k][metric] for k in keys if k in t and metric in t[k]]
    return st.mean(vals) if vals else None


def autodetect():
    cells = {}  # (model, sampling) -> (mtime, dir)
    for tr in PROJECT.glob("logs/train/**/test_results.txt"):
        d = tr.parent
        model, samp = run_meta(d)
        if model is None:
            continue
        key = (model, samp)
        mt = tr.stat().st_mtime
        if key not in cells or mt > cells[key][0]:
            cells[key] = (mt, d)
    return [d for _, d in cells.values()]


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or autodetect()
    rows = []
    for d in dirs:
        tr = d / "test_results.txt"
        if not tr.exists():
            continue
        model, samp = run_meta(d)
        t = parse_test_results(tr)
        rows.append({
            "model": model or d.name, "sampling": "on" if samp else "off",
            "ext_Dice": emean(t, EXT, "Dice"), "ext_Sens": emean(t, EXT, "Sensitivity"),
            "int_Dice": emean(t, INT, "Dice"), "dir": str(d),
        })

    rows = [r for r in rows if r["ext_Dice"] is not None]
    rows.sort(key=lambda r: (r["model"], r["sampling"]))

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else "--"

    print(f"{'model':12s}{'sampling':>10}{'extDice':>9}{'extSens':>9}{'intDice':>9}")
    for r in rows:
        print(f"{r['model']:12s}{r['sampling']:>10}{fmt(r['ext_Dice']):>9}{fmt(r['ext_Sens']):>9}{fmt(r['int_Dice']):>9}")

    # LaTeX
    bestext = max(r["ext_Dice"] for r in rows)
    body = []
    for r in rows:
        ed = f"\\textbf{{{r['ext_Dice']:.3f}}}" if abs(r["ext_Dice"] - bestext) < 1e-9 else f"{r['ext_Dice']:.3f}"
        body.append(f"{r['model']} & {r['sampling']} & {ed} & {fmt(r['ext_Sens'])} & {fmt(r['int_Dice'])} \\\\")
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Controlled fine-tuning baselines (no distillation): SAM and TinyUSFM "
        "trained directly on the task with balanced sampling off vs.\\ on, identical "
        "recipe/seed. External = mean over BUID/BUS\\_UCLM/BUS\\_UCLM\\_filtered; best in \\textbf{bold}.}\n"
        "\\label{tab:baseline_finetune}\n\\begin{tabular}{l c c c c}\n\\toprule\n"
        "Model & Sampling & ext Dice & ext Sens & int Dice \\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    out = PROJECT / "docs" / "kd_baseline_finetune.tex"
    out.write_text(tex)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
