"""Aggregate the teacher-strength × reliability-method sweeps into a 2D view.

Scans the four teacher_strength directions under logs/distill/ and builds a table
of student (TinyUSFM) Dice per (teacher, method), plus the gain of each KD method
over the task-only floor. The central questions:

  * does reliability-KD's gain over vanilla logit-KD shrink as the teacher gets
    stronger?
  * does meta_scalar stay >= logit_kd for every teacher (the "no harm" property)?

Teacher strength (x-axis) is the teacher's own segmentation quality; supply it
with --teacher-dice or let the script read it from the teacher checkpoint file
names (best_epoch_*_dice_<v>.pth) recorded in each run's final_metrics.json
(_meta.best_checkpoint is the *student*, so teacher strength is passed in).

Usage:
  uv run tools/summarize_teacher_strength.py                 # markdown table
  uv run tools/summarize_teacher_strength.py --plot out.png  # + 2D line plot
  uv run tools/summarize_teacher_strength.py \
      --teacher-dice sam=0.764,sam_vit_l=0.83,sam_vit_h=0.82,sam3_teacher=0.86
"""
import argparse
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DISTILL_ROOT = PROJECT_DIR / "logs" / "distill"
STUDENT = "tinyusfm"

# teacher config name → short label, ordered weak→strong (default ordering when
# --teacher-dice is not given).
TEACHERS = {
    "sam": "vit_b",
    "sam_vit_l": "vit_l",
    "sam_vit_h": "vit_h",
    "sam3_teacher": "sam3",
}
METHODS = ["task_only", "logit_kd", "reliability", "learned_pseudo", "meta_scalar"]
BASELINE = "task_only"                      # the no-KD floor every gain is measured against
KD_METHODS = [m for m in METHODS if m != BASELINE]


def _dice(metrics, group):
    node = metrics.get(group)
    if isinstance(node, dict):
        return node.get("Dice")
    return None


def load_direction(teacher, group="teacher_strength"):
    """Return {method: {'int': dice, 'ext': dice}} for one teacher direction.

    ``group`` selects the study bucket segment so the binary
    (``teacher_strength/``) and multiclass (``teacher_strength_mc/``) sweeps —
    both of which live under the same ``{teacher}_to_tinyusfm`` root — are never
    conflated. Exact-segment match: ``/teacher_strength/`` does not match
    ``/teacher_strength_mc/``.
    """
    root = DISTILL_ROOT / f"{teacher}_to_{STUDENT}"
    out = {}
    for fm in root.glob("**/final_metrics.json"):
        if f"/{group}/" not in str(fm):
            continue
        try:
            d = json.loads(fm.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        meta, metrics = d.get("_meta", {}), d.get("metrics", {})
        rn = meta.get("run_name") or ""
        method = rn[len("relab_"):] if rn.startswith("relab_") else rn
        if method not in METHODS:
            continue
        out[method] = {
            "int": _dice(metrics, "internal_mean"),
            "ext": _dice(metrics, "external_mean"),
        }
    return out


def parse_teacher_dice(s):
    if not s:
        return {}
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        out[k.strip()] = float(v)
    return out


def _f(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "  -   "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["int", "ext"], default="ext",
                    help="internal (held-out test) or external (BUID/UCLM) mean Dice")
    ap.add_argument("--teacher-dice", default=None,
                    help="comma list teacher=dice to order/label the strength axis")
    ap.add_argument("--plot", default=None, help="write a 2D line plot to this path")
    ap.add_argument("--group", default="teacher_strength",
                    help="study bucket segment: teacher_strength (binary) | "
                         "teacher_strength_mc (multiclass)")
    args = ap.parse_args()

    tdice = parse_teacher_dice(args.teacher_dice)
    data = {t: load_direction(t, args.group) for t in TEACHERS}
    order = sorted(TEACHERS, key=lambda t: tdice.get(t, 0.0)) if tdice \
        else list(TEACHERS)

    def dice(teacher, method):
        """Student Dice for one (teacher, method) cell; None if the run is missing."""
        return (data[teacher].get(method) or {}).get(args.split)

    # --- markdown table: rows = teacher (weak→strong), cols = method ---
    print(f"\nStudent={STUDENT}  |  split={args.split}-mean Dice\n")
    print("| teacher (Dice) | " + " | ".join(METHODS) + " |")
    print("|" + "---|" * (len(METHODS) + 1))
    for t in order:
        label = TEACHERS[t] + (f" ({tdice[t]:.3f})" if t in tdice else "")
        print(f"| {label} | " + " | ".join(_f(dice(t, m)) for m in METHODS) + " |")

    # --- gain over the no-KD floor + "no-harm" check for meta_scalar ---
    print(f"\nΔDice vs {BASELINE}  (meta_scalar 'no-harm' = meta_scalar ≥ logit_kd)\n")
    print("| teacher | " + " | ".join(KD_METHODS) + " | meta≥logit? |")
    print("|" + "---|" * (len(KD_METHODS) + 2))
    for t in order:
        base = dice(t, BASELINE)
        row = [
            f"{v - base:+.4f}" if v is not None and base is not None else "  -  "
            for v in (dice(t, m) for m in KD_METHODS)
        ]
        lk, ms = dice(t, "logit_kd"), dice(t, "meta_scalar")
        if lk is None or ms is None:
            noharm = "-"
        else:
            noharm = "yes" if ms >= lk else "NO"
        print(f"| {TEACHERS[t]} | " + " | ".join(row) + f" | {noharm} |")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [tdice.get(t, i) for i, t in enumerate(order)]
        plt.figure(figsize=(7, 5))
        for m in METHODS:
            plt.plot(xs, [dice(t, m) for t in order], marker="o", label=m)
        plt.xlabel("teacher strength (Dice)" if tdice else "teacher (weak→strong)")
        plt.ylabel(f"student {args.split}-mean Dice")
        plt.title("Teacher strength × reliability method")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"\n[plot] wrote {args.plot}")


if __name__ == "__main__":
    main()
