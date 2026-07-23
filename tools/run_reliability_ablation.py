"""Dispatch the reliability-KD ablation across local + remote GPU workers.

Reads `config/sweeps/reliability_ablation.yaml`, expands each experiment into a
`distill.py` invocation (base overrides + `method.name=relab_<name>` + the
experiment overrides + `hardware.gpu_ids=[<gpu>]`), and runs them across a pool
of workers. A worker is a `(host, gpu)` pair; `host="local"` runs here,
`host="gpu4"` runs over SSH (shared filesystem, `uv run` available).

Examples:
    # Print every command without running anything.
    uv run tools/run_reliability_ablation.py --dry-run

    # Verify one experiment end-to-end on a tiny 1-epoch run.
    uv run tools/run_reliability_ablation.py --smoke

    # Full sweep across local GPU 1,2 + gpu4 GPU 0-3.
    uv run tools/run_reliability_ablation.py

    # Subset / custom workers.
    uv run tools/run_reliability_ablation.py --only b0_confidence,b4_student_bypass \
        --workers local:1,local:2
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_DIR / "config" / "sweeps" / "reliability_ablation.yaml"
DEFAULT_WORKERS = ["local:1", "local:2", "gpu4:0", "gpu4:1", "gpu4:2", "gpu4:3"]
CONFIG_NAME = "distill_sam_to_usfm_binary"

# Map an experiment-name prefix to its ablation axis (used as a W&B tag so runs
# stay filterable by axis in the UI).
_AXIS_BY_PREFIX = {
    "base_": "baseline",
    "b0_": "factor", "b1_": "factor", "b2_": "factor", "b3_": "factor",
    "lo_": "leave_one_out",
    "tg_": "gate_weight", "sb_": "gate_weight",
    "temp_": "temperature",
    "bs_": "batch_size",
    # teacher_lora sweep (main methods only)
    "task_": "main_method", "logit_": "main_method", "reliability": "main_method",
}


def _axis_tag(name: str) -> str:
    # Sampled-sweep experiments share the non-sampled names under a `smpl_`
    # prefix (so run_name / resume detection stay distinct); tag by the same axis.
    if name.startswith("smpl_"):
        name = name[len("smpl_"):]
    for prefix, axis in _AXIS_BY_PREFIX.items():
        if name.startswith(prefix):
            return axis
    return "misc"


def _extract_exp_dir(logfile: Path):
    """Pull the run's experiment directory from its captured log, or None."""
    try:
        for line in logfile.read_text(errors="ignore").splitlines():
            m = re.search(r"Experiment directory: (\S+)", line)
            if m:
                return m.group(1)
    except OSError:
        pass
    return None


def _teacher_student(manifest):
    """Resolve teacher/student for this manifest from base_overrides.

    Falls back to the distill-config defaults (teacher=sam, student=tinyusfm)
    when a base override is absent. Used to scope resume detection so the same
    experiment name in a different direction is not mistaken for completed.
    """
    teacher, student = "sam", "tinyusfm"
    for ov in manifest.get("base_overrides", []):
        if ov.startswith("teacher="):
            teacher = ov.split("=", 1)[1]
        elif ov.startswith("student="):
            student = ov.split("=", 1)[1]
    return teacher, student


def _completed_experiments(teacher, student, group):
    """Set of experiment names already finished for this teacher->student->group.

    A run counts as completed when its experiment directory contains a
    ``final_metrics.json`` (written only after final evaluation). Two sources
    are scanned and unioned:
      * ``logs/distill/<teacher>_to_<student>/<group>/*/final_metrics.json``
        — authoritative ``run_name`` is read from the file's ``_meta``.
      * prior ``logs/reliability_ablation/*/index.jsonl`` rows with
        ``status == OK`` whose ``exp_dir`` matches this direction AND group.

    The completion key is scoped by the study ``group`` (e.g.
    ``teacher_strength_mc/sam3``), not just the ``teacher_to_student`` direction.
    Two study buckets can share a direction (``teacher_strength`` vs
    ``teacher_strength_mc`` both under ``sam_to_tinyusfm``) and reuse experiment
    names (``base_``, ``logit_kd``); without the group scope a completion in one
    bucket wrongly skipped the same-named experiment in another. Mirrors the
    exact-segment guard in tools/summarize_teacher_strength.py.
    """
    import os
    done = set()
    direction = f"{teacher}_to_{student}"
    group_seg = f"{os.sep}{group}{os.sep}" if group else None

    # Recursive: a manifest's group_label nests one (or more) directory levels
    # under `direction` (see utils.wandb_utils.build_experiment_dir), so a
    # flat `*/` glob would miss runs launched under a group.
    distill_root = PROJECT_DIR / "logs" / "distill" / direction
    for fm in distill_root.glob("**/final_metrics.json"):
        # Only this study's bucket: the run dir is <direction>/<group>/<run>/.
        if group_seg and group_seg not in f"{os.sep}{fm.parent.parent}{os.sep}":
            continue
        try:
            rn = json.loads(fm.read_text()).get("_meta", {}).get("run_name") or ""
        except (OSError, json.JSONDecodeError):
            continue
        if rn.startswith("relab_"):
            done.add(rn[len("relab_"):])

    for ix in (PROJECT_DIR / "logs" / "reliability_ablation").glob("*/index.jsonl"):
        for line in ix.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            exp_dir = rec.get("exp_dir") or ""
            if rec.get("status") != "OK" or direction not in exp_dir:
                continue
            if group_seg and group_seg not in f"{os.sep}{exp_dir}{os.sep}":
                continue
            done.add(rec.get("experiment"))
    return done

# Overrides forced on every run in --smoke mode for a fast end-to-end check.
# train and test must be DISJOINT datasets — the train/test leakage guard
# (utils.data_processing.build_dataset) rejects overlap, and a smoke that reused
# B for both now fails at construction. B (train) + BUID (external test) are both
# small, so the wiring check stays fast.
SMOKE_OVERRIDES = [
    "data.train=[B]",
    "data.test=[BUID]",
    "training.num_epochs=1",
    "training.warmup_epochs=0",
    "wandb.disabled=true",  # key now provided by the `wandb` config group
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--config-name", default=CONFIG_NAME, help="Hydra config name for distill.py")
    p.add_argument("--workers", default=",".join(DEFAULT_WORKERS),
                   help="comma list of host:gpu (host in {local,<ssh-host>})")
    p.add_argument("--only", default=None, help="comma list of experiment names to run")
    p.add_argument("--group", default=None,
                   help="W&B group shared by every run in the sweep; defaults to "
                        "the manifest's top-level `group_label` field (convention: "
                        "'{topic}/{key_param}', e.g. 'reliability_kd/factor_ablation')")
    p.add_argument("--extra", default="", help="extra Hydra overrides appended to every run")
    p.add_argument("--resume", action="store_true",
                   help="skip already-finished experiments (final_metrics.json present) "
                        "and continue partially-trained ones from checkpoints/last.pth "
                        "(passes resume=auto to each launched run)")
    p.add_argument("--dry-run", action="store_true", help="print commands, do not execute")
    p.add_argument("--smoke", action="store_true",
                   help="tiny 1-epoch run; defaults to first 1-2 experiments unless --only given")
    p.add_argument("--poll", type=float, default=5.0, help="scheduler poll interval (s)")
    return p.parse_args()


def build_overrides(manifest, name, exp_overrides, gpu, extra, smoke, group, resume=False):
    overrides = list(manifest.get("base_overrides", []))
    # Experiment label lives in run_name (not method.name) so it can never flip
    # distiller semantics (e.g. _is_online checks method.name=='online').
    overrides.append(f"run_name=relab_{name}")
    overrides += list(exp_overrides)
    overrides.append(f"hardware.gpu_ids=[{gpu}]")
    # Systematic W&B bookkeeping: shared group + per-run axis/name tags.
    if group:
        overrides.append(f"wandb.group={group}")
    overrides.append(f"wandb.tags=[relab,{name},{_axis_tag(name)}]")
    # Continue a partially-trained run from its checkpoints/last.pth (the trainer
    # matches by run_name + hparam signature); a fresh experiment just starts.
    if resume:
        overrides.append("resume=auto")
    if smoke:
        overrides += SMOKE_OVERRIDES
    if extra:
        overrides += shlex.split(extra)
    return overrides


def build_command(host, gpu, overrides, config_name=CONFIG_NAME):
    """Return (argv_list, is_shell_string) for the given worker."""
    base = ["uv", "run", "distill.py", f"--config-name={config_name}", *overrides]
    if host == "local":
        return base, False
    # Remote: run through a login shell so `uv` is on PATH; cd to the shared repo.
    remote = f"cd {shlex.quote(str(PROJECT_DIR))} && " + " ".join(shlex.quote(a) for a in base)
    return ["ssh", host, f"bash -lc {shlex.quote(remote)}"], True


def load_manifest(path):
    """Read a sweep manifest, merging in its `extends:` parent if it has one.

    Sweeps that vary one axis (e.g. the teacher) while holding the method set
    fixed declare `extends: <sibling>.yaml` and state only what differs. Without
    this, comparing N teachers means N copies of the same `experiments:` block,
    and a method added to only N-1 of them silently yields a non-comparable cell
    in the results table.

    Merge rules: `base_overrides` is the parent's list followed by the child's
    (Hydra takes the last occurrence, so the child wins on any repeated key);
    `experiments` is a dict update; any other key the child sets wins outright.
    """
    path = Path(path)
    manifest = yaml.safe_load(path.read_text())
    parent = manifest.pop("extends", None)
    if not parent:
        return manifest

    merged = load_manifest(path.parent / parent)
    overrides = list(merged.get("base_overrides", [])) + list(manifest.pop("base_overrides", []))
    experiments = {**merged.get("experiments", {}), **manifest.pop("experiments", {})}
    merged.update(manifest)
    merged["base_overrides"] = overrides
    merged["experiments"] = experiments
    return merged


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)
    experiments = manifest["experiments"]

    group = args.group or manifest.get("group_label")
    if not group:
        sys.exit(
            f"No W&B group: add a top-level `group_label: {{topic}}/{{key_param}}` "
            f"field to {args.manifest}, or pass --group explicitly."
        )

    names = list(experiments.keys())
    if args.only:
        wanted = [n.strip() for n in args.only.split(",")]
        missing = [n for n in wanted if n not in experiments]
        if missing:
            sys.exit(f"Unknown experiment(s): {missing}\nAvailable: {names}")
        names = wanted
    elif args.smoke:
        names = names[:2]

    if args.resume:
        teacher, student = _teacher_student(manifest)
        done = _completed_experiments(teacher, student, group)
        skipped = [n for n in names if n in done]
        names = [n for n in names if n not in done]
        if skipped:
            print(f"[resume] skipping {len(skipped)} finished: {', '.join(skipped)}")
        print(f"[resume] launching {len(names)} (partial runs continue from last.pth)")
        if not names:
            print("[resume] nothing to run — all experiments already finished.")
            return

    workers = [tuple(w.split(":", 1)) for w in args.workers.split(",") if w]
    workers = [(h, g) for h, g in workers]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_DIR / "logs" / "reliability_ablation" / ts

    # Pre-build every job's command.
    jobs = []
    for name in names:
        # gpu is filled in at dispatch time; placeholder via worker assignment.
        jobs.append((name, experiments[name] or []))

    print(f"Experiments ({len(jobs)}): {', '.join(n for n, _ in jobs)}")
    print(f"Workers: {workers}")
    print(f"Logs: {log_dir}\n")

    if args.dry_run:
        # Don't create the log dir for a dry run (avoids littering empty dirs).
        for name, ov in jobs:
            host, gpu = workers[0]
            overrides = build_overrides(manifest, name, ov, gpu, args.extra, args.smoke, group, args.resume)
            cmd, _ = build_command(host, gpu, overrides, args.config_name)
            printable = cmd if host == "local" else cmd
            print(f"[{name}] ({host}:{gpu})\n  " + " ".join(shlex.quote(c) for c in printable) + "\n")
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    index_path = log_dir / "index.jsonl"
    pending = list(jobs)
    running = {}  # worker_idx -> (name, Popen, logfile_handle)
    running_meta = {}  # worker_idx -> (host, gpu)
    free = list(range(len(workers)))
    results = {}

    while pending or running:
        # Assign pending jobs to free workers.
        while pending and free:
            widx = free.pop(0)
            host, gpu = workers[widx]
            name, ov = pending.pop(0)
            overrides = build_overrides(manifest, name, ov, gpu, args.extra, args.smoke, group, args.resume)
            cmd, _ = build_command(host, gpu, overrides, args.config_name)
            lf = open(log_dir / f"{name}.log", "w")
            lf.write("CMD: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
            lf.flush()
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_DIR), stdout=lf,
                                    stderr=subprocess.STDOUT)
            running[widx] = (name, proc, lf)
            running_meta[widx] = (host, gpu)
            print(f"START [{name}] on {host}:{gpu} (pid {proc.pid})")

        # Reap finished workers.
        for widx, (name, proc, lf) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                lf.close()
                results[name] = rc
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                host, gpu = running_meta[widx]
                exp_dir = _extract_exp_dir(log_dir / f"{name}.log")
                with open(index_path, "a") as ix:
                    ix.write(json.dumps({
                        "run_name": f"relab_{name}", "experiment": name,
                        "exp_dir": exp_dir, "status": status,
                        "host": host, "gpu": gpu,
                    }) + "\n")
                print(f"DONE  [{name}] {status}  -> {log_dir / (name + '.log')}")
                del running[widx]
                free.append(widx)
        time.sleep(args.poll)

    print("\n=== Summary ===")
    ok = sum(1 for v in results.values() if v == 0)
    for name, rc in results.items():
        print(f"  {name}: {'OK' if rc == 0 else f'FAIL(rc={rc})'}")
    print(f"{ok}/{len(results)} succeeded. Logs in {log_dir}")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
