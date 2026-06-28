"""Validate the structured-config schemas against the real config files.

Run: PYTHONPATH=. uv run python tests/test_structured_config.py

Checks, for every entry config that uses the data/training groups:
  1. the composed `data`/`training` nodes merge cleanly onto the typed schema
     (i.e. no unknown keys / type mismatches), and
  2. merging onto the schema does NOT change any value (behaviour preserved).
"""

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config.schema import (  # noqa: E402
    DataConfig,
    TrainingConfig,
    WandbConfig,
    register_schemas,
)
from utils.wandb_utils import resolve_wandb_identity  # noqa: E402

# entry config name -> which groups it carries
ENTRY_CONFIGS = {
    "distill_sam_to_usfm_binary": ("data", "training", "wandb"),
    "distill_usfm_to_sam_binary": ("data", "training", "wandb"),
    "distill": ("data", "training", "wandb"),
    "train_sam": ("data", None, "wandb"),  # train_sam pulls training via model/sam
}

SCHEMAS = {"data": DataConfig, "training": TrainingConfig, "wandb": WandbConfig}


def _merge_check(group, node):
    """Merge `node` onto its schema; return (ok, error_or_diff)."""
    schema = OmegaConf.structured(SCHEMAS[group])
    try:
        merged = OmegaConf.merge(schema, node)
    except Exception as e:  # structured-config rejection
        return False, f"{type(e).__name__}: {e}"
    # value-equivalence: every key present in the raw node is unchanged
    raw = OmegaConf.to_container(node, resolve=False)
    out = OmegaConf.to_container(merged, resolve=False)
    diffs = _diff(raw, out)
    return (not diffs), diffs


def _diff(a, b, prefix=""):
    """Keys present in `a` whose value differs in `b`."""
    out = []
    if isinstance(a, dict):
        for k, v in a.items():
            out += _diff(v, (b or {}).get(k), f"{prefix}.{k}")
    else:
        if a != b:
            out.append(f"{prefix}: {a!r} != {b!r}")
    return out


def main():
    register_schemas()
    failures = []
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "config")):
        for name, groups in ENTRY_CONFIGS.items():
            cfg = compose(config_name=name)
            for group in groups:
                if group is None:
                    continue
                if group not in cfg:
                    failures.append(f"{name}: missing group '{group}'")
                    continue
                ok, info = _merge_check(group, cfg[group])
                status = "OK" if ok else "FAIL"
                print(f"[{status}] {name}.{group}")
                if not ok:
                    failures.append(f"{name}.{group}: {info}")

        # --- wiring tests: dynamic.yaml now references base_schema in its
        # defaults, so compose itself exercises validation. ---
        # 1. valid CLI-style override is applied (yaml stays the value source).
        cfg = compose(config_name="distill_sam_to_usfm_binary",
                      overrides=["data.num_classes=1"])
        if cfg.data.num_classes != 1:
            failures.append("wiring: valid override data.num_classes=1 not applied")
        else:
            print("[OK] wiring: valid override applied (data.num_classes=1)")

        # 2. wrong-type override is rejected at compose time by the schema.
        try:
            compose(config_name="distill_sam_to_usfm_binary",
                    overrides=["data.num_classes=notanint"])
            failures.append("wiring: bad-type override NOT rejected")
        except Exception as e:
            print(f"[OK] wiring: bad-type override rejected ({type(e).__name__})")

        # --- wandb identity + run_reliability_ablation.py override style ---
        # The runner appends wandb.group / wandb.tags / wandb.disabled as plain
        # (non-'+') overrides now that the wandb group provides those keys.
        cfg = compose(
            config_name="distill_sam_to_usfm_binary",
            overrides=[
                "data.train=[B,BUSBRA]",
                "run_name=relab_smpl_reliability",
                "wandb.group=reliability_teacher_lora",
                "wandb.tags=[relab,smpl_reliability]",
                "wandb.disabled=true",
            ],
        )
        ident = resolve_wandb_identity(cfg, default_job_type="distill")
        checks = {
            "project": ident["project"] == "medfm-distill",
            "explicit group honored": ident["group"] == "reliability_teacher_lora",
            "run_name in name": "relab_smpl_reliability" in (ident["name"] or ""),
            "explicit tags kept": "relab" in ident["tags"],
            "auto tags appended": "distill" in ident["tags"] and "B" in ident["tags"],
            "disabled->mode": ident["mode"] == "disabled",
        }
        for label, ok in checks.items():
            print(f"[{'OK' if ok else 'FAIL'}] wandb identity: {label}")
            if not ok:
                failures.append(f"wandb identity: {label} (got {ident})")

        # auto group when none set: "{job}/{method}/{datasets}"
        cfg2 = compose(config_name="distill_sam_to_usfm_binary",
                       overrides=["data.train=[BUSBRA]"])
        g = resolve_wandb_identity(cfg2, "distill")["group"]
        ok = g == "distill/unified/BUSBRA"
        print(f"[{'OK' if ok else 'FAIL'}] wandb identity: auto group = {g}")
        if not ok:
            failures.append(f"wandb auto group: got {g}")

    print("\n" + ("=" * 50))
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ALL STRUCTURED-CONFIG CHECKS PASSED")


if __name__ == "__main__":
    main()
