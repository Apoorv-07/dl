# Member 3 integration patch — pushing checklist

This overlay fixes the source-code integration seams for the 2026-10-28 Member-2 package.
It contains source/config/doc changes only; checkpoint, records and graph binaries are not included.

## What this patch changes

- Normalises the package's `memory_train` / `calibration` split names and `0` / `1` labels into
  Member 3's canonical split and label values without changing sample IDs or embedding vectors.
- Supports Member 2's CSV export (`h_0` through `h_127`) as well as its NPZ export.
- Adds `member2_artifacts/model_adapter.py`, which wraps the real Member 2 adapter in the interface
  Member 3 expects. It loads `member2/training/model_adapter.py` and the actual local checkpoint.
- Handles missing source timestamps safely. For real data, it preserves source timestamps/index order
  instead of using the synthetic attack-burst ordering that remains available only for synthetic tests.
- Handles pipe-separated attack-family metadata as multiple family memberships for family metrics.
- Makes graph files optional for cached-embedding scoring and permits native `.pt` graph files when
  they are supplied. Graph-based explanations/Fidelity+ are marked NOT RUN if graphs are absent.
- Updates configuration/report labels so this package is not falsely described as cross-dataset
  evaluation while its memory dataset is identified only by the generic `dataset1_graph_pt` ID.
- Records the known Member-2 model limitation: reported ROC-AUC 0.36 for `h`; no sign flip is applied.

## Local files required (do not commit to GitHub)

Copy from the Member-2 export folder into `member3/member2_artifacts/`:

- `best_model.pt` renamed to `checkpoint.pt`
- `model_config.yaml`
- `feature_schema.json`
- `preprocessing.json`
- `member2_records.npz`

The `member2_artifacts/.gitignore` in this overlay excludes model/data binaries.

## Install local artifacts and run from the `member3/` project directory (PowerShell)

From the combined repository layout (`member2/` and `member3/` are siblings), run:

```powershell
.\tools\setup_member2_artifacts.ps1 -Member2Package "..\member2\member3_package_20261028_033449"
```

The script copies the records, model config/schema/preprocessing files, `sample_pairs.pt` (if present),
and the checkpoint into `member2_artifacts/`. Large data/checkpoint files remain git-ignored.

Then run:

```powershell
python -m pytest -q
python -c "import numpy as np; z=np.load('member2_artifacts/member2_records.npz'); print(z.files); print(z['embedding'].shape)"
python run_experiment.py --config config.yaml --stage calibration
python run_experiment.py --config config.yaml --stage evaluation
python run_experiment.py --config config.yaml --stage xai
```

To review the exact source changes before pushing:

```powershell
git status --short
git diff --check
git diff
```

If calibration fails, stop and inspect the exact error before continuing. Evaluation uses cached `h`
embeddings; it does not retrain the encoder. With no graph files, XAI/Fidelity+ should be reported as
NOT RUN. If Member 1 supplies matching graph files, put them beside `member2/` and populate
`data.graph_paths` in `config.yaml` with the correct relative paths.

## Before publishing results

The current package reports ROC-AUC 0.36 for the current anomaly score on the validation split.
This patch does not make the detector accurate. Do not flip the score sign. Do not claim cross-dataset
zero-day performance until Member 1 confirms the real Dataset 1 identity and a genuinely independent
test dataset is used. The config intentionally disables the cross-dataset claim.
