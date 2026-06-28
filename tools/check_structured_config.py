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

from config.schema import DataConfig, TrainingConfig, register_schemas  # noqa: E402

# entry config name -> which groups it carries
ENTRY_CONFIGS = {
    "distill_sam_to_usfm_binary": ("data", "training"),
    "distill_usfm_to_sam_binary": ("data", "training"),
    "distill": ("data", "training"),
    "train_sam": ("data", None),  # train_sam pulls training via model/sam
}

SCHEMAS = {"data": DataConfig, "training": TrainingConfig}


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

    print("\n" + ("=" * 50))
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ALL STRUCTURED-CONFIG CHECKS PASSED")


if __name__ == "__main__":
    main()
