"""Regenerate docs/figures/why_lora_better.png.

Panel (a): teacher reliability factors (same student, BUID), FT vs LoRA teacher.
Panel (b): student external-mean Dice for each KD method, by teacher, vs the
           teacher-independent task-only baseline.

Values are sourced from docs/rel_ko.md (§1 main comparison, §2 factor means).
Run: uv run tools/plot_why_lora_better.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FT = "#e1664f"      # full-FT teacher (coral)
LORA = "#2e9c8e"    # LoRA teacher (teal)
OUT = Path(__file__).resolve().parent.parent / "docs" / "figures" / "why_lora_better.png"

# ---- Panel (a): teacher reliability factors over BUID pixels (same student) ----
factors = ["confidence\n(max-prob)", "entropy\npenalty", "correctness gate\n(=pixel acc)"]
a_ft = [0.82, 0.52, 0.94]
a_lora = [0.81, 0.48, 0.91]

# ---- Panel (b): student external-mean Dice, by KD method and teacher ----
methods = ["Logit-KD", "Uncertainty-KD", "Reliability-KD"]
b_ft = [0.647, 0.656, 0.646]     # FT-sampled teacher (0.682); §1 fair-comparison sweep
b_lora = [0.665, 0.628, 0.641]   # LoRA-sampled teacher (0.720); §1 fair-comparison sweep
task_only = 0.630                # teacher-independent task-only baseline (sampling-on)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))
w = 0.38

# Panel (a)
x = np.arange(len(factors))
ax1.bar(x - w / 2, a_ft, w, color=FT, label="FT teacher (0.68)")
ax1.bar(x + w / 2, a_lora, w, color=LORA, label="LoRA teacher (0.72)")
for xi, (vf, vl) in enumerate(zip(a_ft, a_lora)):
    ax1.text(xi - w / 2, vf + 0.01, f"{vf:.2f}", ha="center", va="bottom", fontsize=9)
    ax1.text(xi + w / 2, vl + 0.01, f"{vl:.2f}", ha="center", va="bottom", fontsize=9)
ax1.set_xticks(x)
ax1.set_xticklabels(factors)
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("mean over BUID pixels")
ax1.set_title("(a) Teacher reliability factors (same student, BUID)")
ax1.legend(loc="upper center")

# Panel (b)
x = np.arange(len(methods))
ax2.bar(x - w / 2, b_ft, w, color=FT, label="Logit/Unc/Rel — FT teacher")
ax2.bar(x + w / 2, b_lora, w, color=LORA, label="Logit/Unc/Rel — LoRA teacher")
for xi, (vf, vl) in enumerate(zip(b_ft, b_lora)):
    ax2.text(xi - w / 2, vf + 0.005, f"{vf:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.text(xi + w / 2, vl + 0.005, f"{vl:.3f}", ha="center", va="bottom", fontsize=8)
ax2.axhline(task_only, ls="--", color="0.3",
            label=f"Task-only (no KD)\nteacher-independent: {task_only:.3f}")
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.set_ylim(0, 0.72)
ax2.set_ylabel("external-mean Dice")
ax2.set_title("(b) KD vs the no-KD baseline, by teacher")
ax2.legend(loc="lower right", fontsize=8)

fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"saved {OUT}")
