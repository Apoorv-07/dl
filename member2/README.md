# Self-Supervised Temporal Graph Encoder for Network Intrusion Detection (Member 2)

Benign-only, self-supervised pretraining of a temporal graph encoder on NetFlow data, plus the evaluation tooling
used to test whether its embeddings separate attack traffic from benign traffic.

> **Status (read this first):** the pipeline is complete and reproducible, and the validation data is now
> verified to be on the same scale as the training data. **However, the embeddings do not yet detect attacks.**
> On the corrected validation file the benign-reference kNN score ranks attacks as *more normal* than benign
> (ROC-AUC 0.36 for `h`, 0.43 for `z`), and an *untrained* encoder scores the same. See
> [Results](#results-and-what-you-can-claim) before building anything on top of this.

---

## 1. What this repository contains

| Component | Description |
|---|---|
| Model | GATv2 (2 layers) → global mean pooling per snapshot → Temporal Transformer → `h` (128-d, unit norm) → projection head → `z` (64-d) |
| Pretraining | NT-Xent contrastive loss (temperature 0.1) on **benign-only** sequences from Dataset 1 (`graph.pt`) |
| Anomaly scoring | Benign reference memory + cosine kNN distance (K = 5) |
| Evaluation | `graph_validation_v2.pt` (Dataset 2), with leakage-free thresholding, controls, and diagnostics |

### Architecture

```
snapshot t (t = 1..5):  node features x [N,53], edge_index, edge_attr [E,53]
   └─ GATv2Conv(53→4×128 heads, edge_dim=53) + ELU
   └─ GATv2Conv(512→128, 1 head, edge_dim=53)
   └─ global mean pooling over nodes            → [1,128] per snapshot
stack 5 snapshots                                → [1,5,128]
   └─ + learned positional embedding (1,5,128)
   └─ TransformerEncoder (2 layers, 4 heads, FFN 512, GELU, dropout 0.1)
   └─ mean over 5 tokens → LayerNorm → L2-normalise  → h  [128]   (the embedding to use downstream)
   └─ Linear(128,128) → GELU → Linear(128,64)         → z  [64]    (used only by the contrastive loss)
```

The analytic parameter count is about 644,032. Verify it yourself with
`sum(p.numel() for p in model.parameters())`; do not quote it without checking.

### Training setup

Adam, lr 1e-4, batch size 32, max 100 epochs, early stopping (patience 10, min delta 1e-4), 90/10 train/validation
split of `graph.pt` (seed 42 → 2,220 / 247 sequences). Augmentations: node-feature masking 0.15, per-channel edge-feature
masking 0.15, edge dropout 0.5. Features are clamped to [-5, 5] before the encoder. No scheduler, weight decay or gradient clipping.

Recorded run: best epoch 36, best validation loss 0.11683 (batch 32). **Contrastive loss values are not comparable across
batch sizes** (chance level is ln(2B−1): 4.14 for B = 32, 1.95 for B = 4).

---

## 2. Repository layout

```
models/        gatv2_encoder.py  global_pooling.py  temporal_transformer.py  projection_head.py  member2_model.py
losses/        nt_xent.py
training/
  augmentations.py            two-view augmentation
  train_with_validation.py    original training script (writes checkpoints_validation/)
  train_exp.py                experiment training with switchable options (writes checkpoints_exp/<name>/)
  exp_common.py               shared model builder / encoder / embedding helpers  <-- use this to load checkpoints
  eval_exp.py                 evaluation of any checkpoint, or an untrained control, on graph_validation_v2.pt
  verify_new_validation.py    checks a validation file (scale, shift, duplicates) then evaluates
  diagnose_attack_signal.py   dilution test, size control, linear probes, raw-feature baseline
  score_readouts.py           14 alternative anomaly scorers on saved embeddings (CPU only)
  compare_checkpoints.py      apples-to-apples checkpoint comparison on the held-out graph.pt split
  diagnose_scale.py           per-feature scale table between two files
checkpoints_validation/        best_model.pt  training_history.json   (not stored in git, see §7)
diagnostics/                   JSON / NPZ outputs written by the scripts above
graph.pt, graph_validation_v2.pt    data (not stored in git, see §7)
```

---

## 3. Setup

Tested on Windows, Python 3.13, PyTorch 2.11 (CUDA 12.8), PyTorch Geometric 2.8, RTX 4060 laptop GPU.
A GPU is strongly recommended for training; evaluation also runs on CPU, just slower.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torch_geometric scikit-learn scipy numpy
```

---

## 4. Data

Both files are `torch.save`d dicts with a `"sequences"` list. Each sequence has `"snapshots"` (5 PyG `Data` objects with
`x [N,53]`, `edge_index`, `edge_attr [E,53]`, and for validation `edge_y`) and, for validation, a window-level `"label"`
(0 benign / 1 attack) and `"attack_types"`.

| File | Role | Contents |
|---|---|---|
| `graph.pt` | Dataset 1: pretraining | 2,467 benign sequences, 5 snapshots, 53 features, all features in [0, 1] |
| `graph_validation_v2.pt` | Dataset 2: evaluation | 2,463 sequences: 1,448 benign, 1,015 attack (14 attack types) |

**Feature scale matters.** The first `graph_validation.pt` was on different scales from `graph.pt` for 36 of 53
features (for example `TCP_FLAGS` max 90, `ICMP_TYPE` max 12,235), which made it out of distribution
(shift AUC 0.995) and produced misleading, collapsed embeddings. `graph_validation_v2.pt` fixes this: 0/53 flagged
features, shift AUC 0.525. **Always run `verify_new_validation.py` on any new validation file before trusting a result.**

Known dataset properties (these affect any score you compute):

- **Graph size is a strong confound.** Size alone separates the classes (size-only AUC 0.84): benign windows have a median of
  6,942 edges, attack windows 9,758.
- **8.6 % of benign validation windows are nearly empty** (fewer than 30 nodes).
- **Window labels are loose.** A window is "attack" if it contains any attack flow; the median attack window is only
  about 2.2 % attack flows.

---

## 5. How to run

All commands run from the project root with the virtual environment active.

```powershell
# 1. Check a validation file (scale, shift vs graph.pt, duplicates), then evaluate if it passes
python training\verify_new_validation.py --val "graph_validation_v2.pt" *> verify_console.txt

# 2. Train the original model (does NOT overwrite if you change CHECKPOINT_DIR; it writes checkpoints_validation\)
python training\train_with_validation.py

# 3. Evaluate any checkpoint (same protocol as step 1)
python training\eval_exp.py --label baseline --ckpt checkpoints_validation\best_model.pt

# 4. Control: evaluate an UNTRAINED encoder
python training\eval_exp.py --label random_init --random-init

# 5. Experiments, one change at a time (never overwrites an existing run)
python training\train_exp.py --name e1_log  --transform log
python training\train_exp.py --name e2_pool --pooling meanmax
python training\train_exp.py --name e3_aug  --edge-drop 0.2 --feature-mask 0.10

# 6. Diagnostics on saved embeddings
python training\diagnose_attack_signal.py --val "graph_validation_v2.pt"
python training\score_readouts.py
```

---

## 6. Using the encoder downstream (for Member 3)

Use `h` (128-d, L2-normalised) as the embedding. `z` exists only for the contrastive loss.

```python
import sys, torch
sys.path.append("training")          # exp_common.py lives here
from exp_common import load_checkpoint, embed

model, cfg, ckpt = load_checkpoint("checkpoints_validation/best_model.pt")
sequences = torch.load("graph_validation_v2.pt", weights_only=False)["sequences"]
H, Z = embed(model, sequences, cfg)  # numpy arrays: H [n,128] (unit norm), Z [n,64]
```

Input contract, which must be respected or embeddings are meaningless:

- 5 snapshots per sequence, node and edge features dimension 53, **on the same scale as `graph.pt` (values in [0, 1])**.
- Features are clamped to [-5, 5] inside the encoder code path (`exp_common.py` does this; if you write your own inference
  code, do the same).
- Evaluation mode, no augmentation at inference.

Anomaly scoring protocol (this is the part to keep unchanged):

1. Build the **benign reference memory** from `graph.pt` training embeddings (the first 90 % after the seed-42 permutation).
2. Score = mean cosine distance to the K = 5 nearest memory embeddings.
3. Choose any threshold from **benign data only** (for example the 95th percentile of held-out benign scores).
   Never choose it on test labels.

---

## 7. Results and what you can claim

Evaluation on `graph_validation_v2.pt`: memory = `graph.pt` training embeddings, threshold = benign calibration
95th percentile, test set = 724 benign + 1,015 attack windows (prevalence 0.584). No test labels used for any choice.

| Model | h kNN AUC | z kNN AUC | h linear probe\* | notes |
|---|---|---|---|---|
| Trained baseline (best epoch 36) | 0.360 | 0.427 | 0.849 | AP 0.479 / 0.521; TPR 0.002 / 0.013 at FPR ≈ 0.04 |
| Untrained encoder (control) | 0.355 | 0.357 | 0.866 | same ranking as the trained model |
| e1: log input transform | 0.322 | 0.326 | 0.866 | lower loss (0.0929), worse AUC |
| Raw averaged-feature kNN (baseline) | 0.556 | – | 0.970 | 0.970 on windows with ≥ 50 % attack flows |
| Graph size only (probe)\* | – | – | 0.837 | |

\* Linear probes use labels and random CV folds; consecutive windows may overlap in time, so they are optimistic.
**Diagnostics only, never report them as detection performance.**

What the evidence supports:

- The Dataset 2 scale mismatch is fixed, and the model's benign embeddings are in-distribution (shift AUC ≈ 0.5).
- Size-controlled AUC is still below 0.5, so graph size alone does not explain the result.
- Dilution by loosely labelled windows does not explain it for the embeddings: AUC stays at 0.34–0.52 even on windows with
  at least 50 % attack flows, while raw features reach 0.97.
- No alternative scorer tried (kNN with different K, centroid, Mahalanobis, local outlier factor) gets above 0.5.
- Contrastive training did not add attack-relevant information beyond an untrained network on this evaluation.

**Do not claim** zero-day detection, cross-dataset generalisation, or any detection performance from these embeddings.
**Do not flip the score sign** to turn 0.36 into 0.64; that is a post-hoc choice with no justification.

Experiments e2 (mean + max pooling) and e3 (weaker augmentation) are implemented in `train_exp.py` but have not been
run/reported yet.

---

## 8. Known limitations and open questions

- Mean pooling after the graph encoder may average away rare or extreme flows; max-pooling is untested.
- Benign-only instance discrimination trains the model to separate benign windows from each other, not attacks from benign.
- Size and near-empty windows are confounds that need explicit handling in any evaluation you add.
- Results come from a single trained checkpoint, a single seed and a single validation file; there are no confidence intervals
  across training runs.

---

## 9. Reproducibility and storage notes

- Seeds: training and splits use seed 42. Checkpoints and histories are never overwritten by `train_exp.py`.
- **Data and checkpoints are not stored in git** (`.gitignore` excludes `*.pt`, `diagnostics/`, `venv/`). Share them through
  Git LFS, a GitHub Release asset, or your team's shared drive, and tell Member 3 where they are.
  Required to reproduce: `graph.pt`, `graph_validation_v2.pt`, `checkpoints_validation/best_model.pt`.
- The `feature_cols.pkl` file (optional) gives the 53 feature names for readable reports.