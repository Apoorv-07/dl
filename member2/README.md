# Self-Supervised Temporal Graph Encoder for Network Intrusion Detection (Member 2)

Benign-only, self-supervised pretraining of a temporal graph encoder on NetFlow data, plus the evaluation tooling used to test
whether its embeddings separate attack traffic from benign traffic.

> **Status (read this first).** The pipeline is complete and reproducible, and the validation data is verified to be on the same
> scale as the training data. The recommended model (**e2**, mean+max pooling) ranks attacks only **modestly** above benign:
> ROC-AUC about **0.57-0.60** with the benign-reference kNN scorer, from **0.36** for the first model. It is **not** a reliable
> detector and **no zero-day or cross-dataset claim is supported**. A plain graph-size rule is competitive with, and in
> supervised folds slightly better than, any model here. See [Results](#7-results-and-what-you-can-claim).

---

## 1. What this repository contains

| Component | Description |
|---|---|
| Model | GATv2 (2 layers) -> per-snapshot pooling over nodes -> Temporal Transformer -> `h` (128-d, unit norm) -> projection head -> `z` (64-d) |
| Pretraining | NT-Xent contrastive loss (temperature 0.1) on **benign-only** sequences from Dataset 1 (`graph.pt`) |
| Anomaly scoring | Benign reference memory + cosine kNN distance (K = 5) |
| Evaluation | `graph_validation_v2.pt` (Dataset 2), with leakage-free thresholding, untrained controls, and diagnostics |

### Architecture

```
snapshot t (t = 1..5):  node features x [N,53], edge_index, edge_attr [E,53]
   -> GATv2Conv(53 -> 4 heads x 128, edge_dim=53) + ELU
   -> GATv2Conv(512 -> 128, 1 head, edge_dim=53)
   -> pooling over nodes -> [1,128] per snapshot
        mean pooling (baseline)            : mean over nodes
        mean+max pooling (e2, recommended) : concat(mean, max) -> Linear(256,128)
stack 5 snapshots -> [1,5,128] + learned positional embedding
   -> TransformerEncoder (2 layers, 4 heads, FFN 512, GELU, dropout 0.1, post-LN)
   -> mean over the 5 tokens -> LayerNorm -> L2-normalise   -> h [128]  (the embedding to use downstream)
   -> Linear(128,128) -> GELU -> Linear(128,64)             -> z [64]   (contrastive projection; also usable)
```

Parameters: 644,032 (baseline, verified) and 676,928 (e2, adds the 256->128 pooling layer).

### Training setup (both models)

Adam, lr 1e-4, batch 32, max 100 epochs, early stopping (patience 10, min delta 1e-4), 90/10 train/validation split of `graph.pt`
(seed 42 -> 2,220 / 247 sequences). Augmentations: node-feature masking 0.15, per-channel edge-feature masking 0.15, edge dropout 0.5.
Features are clamped to [-5, 5] before the encoder. Checkpoint selection uses only the benign held-out contrastive loss; no attack
label is used anywhere in pretraining. **Contrastive loss values are not comparable across different architectures or batch sizes.**

---

## 2. Repository layout

```
models/        gatv2_encoder.py  global_pooling.py  temporal_transformer.py  projection_head.py  member2_model.py
losses/        nt_xent.py
training/
  augmentations.py            two-view augmentation
  train_with_validation.py    ORIGINAL training script (writes checkpoints_validation/)
  train_exp.py                experiment training with switchable options, resume, diagnostics (writes checkpoints_exp/<name>/)
  exp_common.py               shared model builder / encoder / embedding helpers  <-- use this to load any checkpoint
  model_adapter.py            encode(), encode_with_attention(), metadata()  (for Member 3)
  export_member3_package.py   builds the hand-off package (records, schema, preprocessing, sample pairs)
  eval_exp.py                 evaluate any checkpoint, or an UNTRAINED encoder, on graph_validation_v2.pt
  verify_new_validation.py    checks a validation file (scale, shift, duplicates), then evaluates
  compare_runs.py             one comparison table from the eval_*.json files
  diagnose_attack_signal.py   dilution test, size control, linear probes, raw-feature baseline
  score_readouts.py           14 alternative anomaly scorers on saved embeddings (CPU only)
  sup_common.py sup_folds.py  SUPERVISED rolling-origin chronological evaluation (CPU)
  finetune_sup.py             SUPERVISED fine-tuning, one fold at a time (GPU)
  sup_probe.py  flow_probe.py single-split supervised probe; flow-level feasibility check
checkpoints_validation/        baseline best_model.pt + training_history.json
checkpoints_exp/e2_pool/       recommended checkpoint
member3_package*/              hand-off packages (baseline and e2); see section 6
diagnostics/                   JSON / NPZ outputs written by the scripts (not stored in git)
graph.pt, graph_validation_v2.pt    data (not stored in git)
```

---

## 3. Setup

Tested on Windows, Python 3.13.7, PyTorch 2.11 (CUDA 12.8), PyTorch Geometric 2.8, RTX 4060 laptop GPU. Evaluation also runs on CPU.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torch_geometric scikit-learn scipy numpy
```

---

## 4. Data

Both files are `torch.save`d dicts with a `"sequences"` list. Each sequence has `sequence_id`, `snapshots` (5 PyG `Data` objects with
`x [N,53]`, `edge_index`, `edge_attr [E,53]`, and for validation `edge_y`), `label`, `feature_list`, `node_id_maps`, `window_size_sec` (30 s).
Validation sequences also carry `attack_types`, `attack_ratio`, `num_flows`, `num_attack_flows`, `start_time`, `end_time`, `window_ids`,
`low_traffic`. Dataset 1 has no timestamps or window ids.

| File | Role | Contents |
|---|---|---|
| `graph.pt` | Dataset 1: pretraining | 2,467 benign sequences (label `normal`), 5 snapshots, 53 features, all in [0, 1] |
| `graph_validation_v2.pt` | Dataset 2: evaluation | 2,463 sequences: 1,448 benign, 1,015 attack (14 attack types), same source dataset (NF-CSE-CIC-IDS2018-v3) |

Member 1 normalises before saving (benign StandardScaler -> log1p -> min-max). The model does not rescale.

**The first `graph_validation.pt` was on different scales from `graph.pt` for 36 of 53 features** (for example `TCP_FLAGS` max 90),
which made it out of distribution (shift AUC 0.995). `graph_validation_v2.pt` fixes this (0/53 flagged features, shift AUC about 0.5).
**Always run `verify_new_validation.py` on any new validation file before trusting a result.**

Dataset properties that affect any score:

- **Graph size is a strong confound.** Size alone separates the classes (AUC 0.84): benign windows have a median of 6,942 edges, attack 9,758.
- **8.6 % of benign validation windows are nearly empty** (fewer than 30 nodes).
- **Window labels are loose.** A window is "attack" if it contains any attack flow; the median attack window is about 2.2 % attack flows.
- **Attack families occur in different periods**, so any time-ordered evaluation tests on families the model has not seen.

---

## 5. How to run

From the project root with the virtual environment active.

```powershell
# check a validation file
python training\verify_new_validation.py --val graph_validation_v2.pt *> verify_console.txt

# original training
python training\train_with_validation.py

# experiments (never overwrite an existing run; add --resume to continue one; use a new --name and --seed for repeats)
python training\train_exp.py --name e2_pool --pooling meanmax
python training\train_exp.py --name e2_pool_s7 --pooling meanmax --seed 7

# evaluate any checkpoint, plus the untrained controls
python training\eval_exp.py --label e2_pool --ckpt checkpoints_exp\e2_pool\best_model.pt
python training\eval_exp.py --label random_meanmax --random-init --pooling meanmax
python training\compare_runs.py

# supervised, chronological (needs a records file from export_member3_package.py)
python training\sup_folds.py --records <package>\member2_records.csv --val graph_validation_v2.pt
```

---

## 6. Using the encoder downstream (for Member 3)

Use `h` (128-d, L2-normalised). **Recommended checkpoint for the unsupervised kNN anomaly scorer:** `checkpoints_exp/e2_pool/best_model.pt`
(SHA-256 `07f9940b032bd970cd8d0657e94c4310588c86a52687a8b94d8a8431b61fc0c0`, best epoch 65). The baseline
`checkpoints_validation/best_model.pt` (SHA-256 `e4535f1e645f30effabb04e4970fd88abda8fcd983b5835ee43abdfca5ef16be`, best epoch 36) is kept for comparison.

```python
import sys, torch
sys.path.append("training")
from model_adapter import Member2Adapter

ad = Member2Adapter("checkpoints_exp/e2_pool/best_model.pt")
seqs = torch.load("graph_validation_v2.pt", weights_only=False)["sequences"]
h = ad.encode(seqs[0]["snapshots"])                       # (128,) unit norm
out = ad.encode_with_attention(seqs[0]["snapshots"])      # h, z, GAT edge attention, temporal attention
print(ad.metadata())
```

Input contract: 5 snapshots, 53 node and 53 edge features **on the `graph.pt` scale (values in [0,1])**, clamped to [-5,5] by the adapter,
evaluation mode, no augmentation. The package folders contain `member2_records.npz/.csv` (embeddings plus `sample_id`, `window_id`,
`timestamp`, `label`, `attack_family`), `sample_pairs.pt`, `model_config.yaml`, `feature_schema.json`, `preprocessing.json` (splits, versions and file SHA-256).

Scoring protocol (keep unchanged): memory = `graph.pt` training embeddings (first 90 % after the seed-42 permutation); score = mean cosine
distance to the K = 5 nearest memory embeddings; thresholds from benign data only, never from test labels.

---

## 7. Results and what you can claim

### 7.1 Unsupervised (the intended design): benign-reference kNN, K = 5

Test set: 724 benign + 1,015 attack windows (prevalence 0.584). No test label was used for any choice.

| Model | `h` AUC | `z` AUC | PR-AUC `h` / `z` | AUC within size bins `h` / `z` | TPR at about 5 % FPR `h` / `z` |
|---|---|---|---|---|---|
| Baseline (mean pooling, epoch 36) | 0.360 | 0.427 | 0.479 / 0.521 | 0.400 / 0.418 | 0.002 / 0.013 |
| Untrained, mean pooling | 0.355 | 0.357 | 0.477 / 0.477 | - | - |
| e1: log input transform | 0.322 | 0.326 | 0.462 / 0.469 | - | - |
| Untrained, mean+max pooling | 0.402 | 0.406 | 0.495 / 0.496 | 0.471 / 0.468 | 0.000 / 0.000 |
| **e2: mean+max, seed 42 (recommended)** | **0.583** | **0.601** | 0.653 / 0.670 | 0.528 / 0.549 | 0.093 / 0.090 |
| e2: mean+max, seed 7 | 0.566 | 0.566 | 0.659 / 0.641 | 0.527 / 0.515 | 0.137 / 0.090 |

- The same architecture with random weights scores about 0.40, so the e2 gain comes from training, not just the new pooling layer.
- Both seeds agree (AUC about 0.57-0.60); report the range, not the better run. The gain is small inside size bins and on windows with at least 20 % attack flows (seed 7: 0.52-0.53).
- Mahalanobis and the other scorers tried stay at or below 0.5. Scores are never sign-flipped.

### 7.2 Supervised, rolling-origin chronological folds (uses attack labels; NOT the intended zero-day design)

Four folds (train on earlier periods, test on the next), threshold chosen on a calibration block only. Mean over folds:

| Features | AUC | Balanced accuracy |
|---|---|---|
| Graph size only | 0.773 | 0.684 |
| Baseline `h` + size | 0.813 | 0.678 |
| e2 `h` + size | 0.757 | 0.623 |
| Raw per-window feature statistics + size | 0.671 | 0.651 |

No supervised variant clearly beats the graph-size rule on balanced accuracy, and generalisation to unseen attack families is weak (fold with brute-force / XSS / SQL injection: AUC about 0.48 for e2).
A sensitivity run that excludes attack windows with under 5 % attack flows (602 of 1,015 removed) makes graph size even stronger (AUC 0.935) and the embeddings no better.
Per-flow classifiers on raw flow features did not generalise across time either (mean flow AUC 0.54-0.60).

### 7.3 What you can and cannot claim

**Supported:** the Dataset 2 scale mismatch is fixed; with mean+max pooling the self-supervised model ranks attacks modestly above chance (AUC 0.57-0.60), and an untrained network of the same architecture does not;
the result repeats across two seeds.

**Not supported:** reliable detection, zero-day detection, cross-dataset generalisation (both datasets come from the same source capture), or any advantage over a graph-size baseline in supervised settings.
Do not flip scores to turn an AUC below 0.5 into one above it, and do not select a checkpoint, scorer or threshold using test labels.
Any explanation work (for example Fidelity+) should state that the underlying score is only weakly informative.

---

## 8. Known limitations and open questions

- Gains are modest and based on two seeds and one validation file; no confidence interval across training runs beyond those two.
- Size and near-empty windows are confounds that need explicit handling in any evaluation you add.
- Window labels are diluted (any attack flow), and several attack families appear only in single periods.
- Untested: mean+max pooling with weaker augmentation, longer or smoother training (the e2 validation loss is noisy: only 247 held-out sequences).

## 9. Reproducibility and storage

- Seeds: training and splits use seed 42 by default (`--seed` changes both the model initialisation and the train/validation split).
- Data files are not stored in git. Share `graph.pt` and `graph_validation_v2.pt` through a shared drive. Checkpoints and package files are small and are committed with `git add -f`.
- `.gitignore` excludes `*.pt`, `*.npz`, `diagnostics/`, `venv/` and console outputs.