# Archived configs

Configs gathered here are **not referenced** by any active entrypoint, runner
script, or doc as of this commit. They are kept (not deleted) for reference and
reproducibility. Original sub-paths are preserved (e.g. `sweeps/`, `model/`).

To reuse one, either move it back or point Hydra at the archived path, e.g.:

```bash
uv run python distill.py --config-name _archive/sweeps/distill_sweep
```

Contents:
- `sweeps/` — legacy / per-GPU / pipeline sweep configs not mentioned in docs or
  code. The actively-documented reliability sweeps remain under `config/sweeps/`.
- `test_sam.yaml` — references the removed `config/train.yaml`, so it no longer
  composes.
- `model/USFM.yaml` — standalone stub superseded by `config/model/usfm.yaml`
  (which extends `tinyusfm`); no entrypoint selects `model=USFM`.
