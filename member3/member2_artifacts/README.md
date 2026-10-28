# Local Member-2 handoff artifacts

Copy these files here locally from Member 2's `member3_package_20261028_033449` folder:

- `checkpoint.pt` (from `checkpoints_validation/best_model.pt`)
- `model_config.yaml`
- `feature_schema.json`
- `preprocessing.json`
- `member2_records.npz`

The checkpoint and records are intentionally ignored by Git. The tracked `model_adapter.py` is a
bridge that imports the real adapter from the sibling `../member2/training/model_adapter.py` in this
monorepo; it does not contain a fake encoder.
