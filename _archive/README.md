# Archived entrypoints

Broken / superseded entrypoints kept for reference (not deleted).

- `test_sam.py` + `train_lightning.py` — both compose Hydra `config_name="train"`
  (or a config whose defaults include `train`), but `config/train.yaml` was
  removed, so neither composes. Supervised training/testing now goes through
  `train.py` (`config_name="train_sam"`, with `mode=test` for evaluation).

To revive one, restore `config/train.yaml` and move the script back to the repo
root (the `@hydra.main` `config_path="config"` is relative to the file).
