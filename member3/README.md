# Member 3 - Inference, Memory, Explainability and Evaluation

Adaptive Memory Bank, incremental FAISS retrieval, confidence-aware threshold, multi-stage alert
verification, explainable graph attribution, cross-dataset / zero-day evaluation, publication
artefacts and a minimal FastAPI service for
*"Adaptive Self-Supervised Temporal Graph Foundation Model for Cross-Dataset Zero-Day Intrusion
Detection"*.

Member 3 is the last stage of the pipeline. It **consumes** Member 1's graphs/splits and Member 2's
embeddings; it does not train, does not build graphs, and does not contain a second model
implementation. This repository is **source only**: no datasets, no mock data, no committed results
or figures. Everything under `results/`, `figures/` and `state/` is produced by a run.

---

## 1. Scope

Owned here:

| Area | Module |
|---|---|
| Adaptive Memory Bank (representative normal embeddings, capacity, online benign updates) | `member3/memory.py` |
| Incremental FAISS HNSW index (+ flat/numpy back-ends for the timing ablation) | `member3/memory.py` |
| Anomaly scoring | `member3/inference.py` |
| Confidence-aware threshold calibration | `member3/inference.py` |
| Multi-stage (consecutive-anomaly) alert verification + detection delay | `member3/inference.py` |
| Online benign memory adaptation (concept drift) | `member3/inference.py` |
| GATv2 attribution orchestration, explanation object | `member3/explain.py` |
| Fidelity+ masking-and-recompute, bootstrap CIs | `member3/explain.py` |
| Cross-dataset + leave-one-attack-family-out evaluation, metrics, tables, figures, manifest | `member3/evaluate.py` |
| upstream handoff (Member 1 + Member 2): adapter loading, record I/O, leakage guards, masking | `member3/contract.py` |
| FastAPI inference service + analyst view | `member3/api.py` |

Explicitly **not** owned: dataset download/parsing, NetFlow preprocessing, adaptive windowing,
temporal graph construction, GATv2 / Temporal Transformer design, NT-Xent or projection-head
training, foundation-model training (Members 1-2), any database/queue/container infrastructure.

## 2. Architecture (implemented data flow)

```
Member-2 embedding z  (frozen checkpoint; produced by GATv2 -> Temporal Transformer -> proj head)
        |  L2-normalise if the adapter declares l2_normalized = false
        v
Adaptive Memory Bank        representative NORMAL embeddings (train_benign split only)
        v
FAISS HNSW k-NN search      inner product on unit vectors == cosine similarity
        v
anomaly score  s = 1 - max_j cos(z, m_j)          (the single scoring definition used everywhere)
        v
confidence-aware threshold  tau, calibrated on the calibration_benign split only
        v
multi-stage verification    consecutive anomaly streak >= N (default 3) -> CONFIRMED ALERT
        v
GATv2 structural attribution  -> top hosts / flows / features
        v
Fidelity+ validation        mask top-K components -> re-encode -> re-score -> % score reduction
        v
evaluation / metrics / statistics / figures / tables / manifest  ->  FastAPI + analyst view
```

Scoring is defined once (§9 of the brief): cosine similarity to the nearest stored normal, turned
into a distance-like anomaly score. No Euclidean path exists anywhere in this code, and
`results/manifest.json` records the neighbour aggregation actually used
(`retrieval.neighbor_aggregation`, default `max`).

## 3. Upstream assumptions (also in `docs/MEMBER2_HANDOFF_CONTRACT.md`)

1. Member 2 exposes `encode()`, `encode_with_attention()` and `metadata()` (see
   [`docs/MEMBER2_HANDOFF_CONTRACT.md`](docs/MEMBER2_HANDOFF_CONTRACT.md)).
2. Embeddings are the post-projection-head representation, L2-normalised (or Member 3 normalises).
3. A records table (`.npz`/`.csv`) carries `dataset_id, split, sample_id, window_id, timestamp,
   embedding, label, attack_family` with four splits: `train_benign`, `calibration_benign`, `test`,
   `online_benign`.
4. Labels exist **only** for evaluation; the memory bank and the threshold never see them.
5. Graph sequences are available if node/edge-masking Fidelity+ is to be measured.
6. Snapshots are 30 s causality-aware windows; `T = 5` per sequence.

If an item is missing, the run aborts or that experiment is reported as `NOT RUN` with the blocking
reason - never estimated, and never substituted with anything Member 3 invented (see §13).

## 4. Required Member-2 artefacts

```
member2_artifacts/
├── checkpoint.pt
├── model_config.yaml
├── preprocessing.json
├── feature_schema.json
├── model_adapter.py          # FoundationModelAdapter
└── sample/                   # sample input/output for the contract self-test
```

Full specification - including the **Member 1** part (graph objects, id maps, snapshot/windowing
spec, split non-overlap proof, stream/episode ordering, preprocessing metadata) - plus tensor shapes,
example calls and per-member validation checklists: **`docs/MEMBER2_HANDOFF_CONTRACT.md`**. Send it to
Members 1 and 2 as-is; it is the only integration document they need.

## 5. Installation

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # numpy pandas scikit-learn matplotlib scipy PyYAML
                                         # faiss-cpu fastapi uvicorn pytest
```

`faiss-cpu` is required for `memory.index_type: hnsw`; without it the run **fails loudly** rather
than silently downgrading the retrieval claim. `torch` / `torch_geometric` are *not* Member-3
dependencies - the Member-2 adapter may import them itself.

## 6. Configuration

One file: `config.yaml` (no env-var layering, no second config system). Every value that can move
a number in the paper lives there and is copied into `results/manifest.json`:
upstream artefact paths, `embedding_dim`, memory capacity/index/selection/replacement, `k`,
neighbour aggregation, threshold strategy + percentile + MAD fence + safety margin, debounce steps,
adaptation policy, evaluation protocol, XAI masking options, bootstrap settings, seed, output dirs.

## 7. Run

```bash
python run_experiment.py --config config.yaml                       # everything
python run_experiment.py --config config.yaml --stage calibration     # memory + threshold -> state/
python run_experiment.py --config config.yaml --stage evaluation      # metrics, latency, zero-day
python run_experiment.py --config config.yaml --stage xai             # attribution + Fidelity+
python run_experiment.py --config config.yaml --stage ablation        # 4 ablation arms
pytest -q                                                             # 35 tests, no datasets needed
```

Stages consume only artefacts saved by earlier stages (`state/memory_index.bin`,
`state/memory_index_vectors.npz`, `state/memory_index_meta.json`, `state/threshold.json`), so a
stage can be re-run alone; `--stage calibration` must exist first. `--fail-fast` aborts on the
first failed stage instead of continuing.

There is **no fallback data and no stand-in model in the pipeline**. With no `data.records_path`, or
no Member-2 adapter file, the run aborts and prints what is missing:

```
ContractError: Member-2 adapter not found at 'member2_artifacts/model_adapter.py'.
Deliver the artefacts specified in docs/MEMBER2_HANDOFF_CONTRACT.md (model_adapter.py +
checkpoint.pt + model_config.yaml) and set member2.artifacts_dir accordingly.
```

That is deliberate: any number produced from substituted data would be a fabricated number. The only
encoder stand-in in this repository lives in `tests/test_member3.py` (`TinyAdapter`, a test double
for the *interface*), it never writes into the repository, and it cannot be selected by
`run_experiment.py`.

## 8. Outputs

```
state/     memory_index.bin, memory_index_vectors.npz, memory_index_meta.json, threshold.json, adapter_meta.json
results/
├── metrics.json               every metric, per stage, machine-readable
├── metrics.csv                flattened metric/value table
├── predictions.csv            one row per window: score, similarity, threshold, confidence, streak,
│                              alert state, delay, memory version, checkpoint, neighbour ids, and the
│                              per-alert attribution columns joined by the xai stage  (single audit table)
├── attack_family_results.csv  per-family performance (family vs benign)
├── ablations.csv              threshold / debounce / memory / retrieval arms
├── latency.csv                per-stage mean, median, p95, p99 + measurement conditions
├── fidelity_plus.csv          per-alert before/after score, reduction, coverage, masking basis
├── paper_tables.csv           tidy long format of every table (metric, value, unit, n, dataset, split, config)
├── tables.md                  the same tables rendered for reading / LaTeX pasting
├── explanations.json          compact explanation object per evaluated alert (full detail)
└── manifest.json              reproducibility manifest: provenance of every stage, failed stages + reasons

figures/   fig01_architecture ... fig12_attack_family_performance (.png, 300 dpi) + figure_status.json
```

Figure numbering follows the paper (the brief's "FIGURE n"); `figures/figure_status.json` maps each
number to `generated` or `NOT RUN: <reason>`:

| file | brief figure |
|---|---|
| `fig01_architecture.png` | FIGURE 1 architecture / inference path |
| `fig02_roc_curve.png` | FIGURE 2 ROC |
| `fig03_pr_curve.png` | FIGURE 3 Precision-Recall |
| `fig04_score_distribution.png` | FIGURE 4 normal vs anomalous scores |
| `fig05_confusion_matrix.png` | FIGURE 5 confusion matrix (alert-level) |
| `fig06_latency_distribution.png` | FIGURE 6 inference latency distribution |
| `fig07_memory_bank_ablation.png` | FIGURE 7 memory adaptation ablation |
| `fig08_threshold_ablation.png` (+ `fig08b_..._fpr.png`) | FIGURE 8 threshold strategy ablation |
| `fig09_debounce_ablation.png` | FIGURE 9 debounce ablation |
| `fig10_fidelity_plus.png` | FIGURE 10 Fidelity+ before/after masking |
| `fig11_cross_dataset_comparison.png` | FIGURE 11 cross-dataset (needs >= 2 test datasets) |
| `fig12_attack_family_performance.png` | FIGURE 12 per-attack-family zero-day |

Tables `TABLE1..TABLE9` (overall cross-dataset, per attack family, leave-one-family-out, latency,
threshold ablation, debounce ablation, memory adaptation, Fidelity+, ablation summary) each carry
sample counts, dataset, split and `config_id`.

## 9. Metrics produced

* **Classification**: precision, recall, F1, ROC-AUC, AUPRC, FPR, FNR - at *sample* level and at
  *alert* (episode) level, always with n.
* **Operational**: per-stage latency (mean/median/p95/p99/min/max) for Member-2 encode, FAISS
  retrieval, thresholding, alert verification (per-window and amortised-batch), Member-3 total;
  throughput; memory-bank size; FAISS configuration; detection delay in windows and seconds
  (window = `data.window_seconds`).
* **Zero-day**: per-attack-family results (family vs benign windows), leave-one-attack-family-out
  folds with a **leakage audit** (`tau_delta_vs_baseline`, `memory_changed`), cross-dataset table.
* **XAI**: Fidelity+ (mean, median, std, bootstrap 95% CI, fraction of score drops, before/after
  score, n alerts), attention vs random vs lowest-attention masking, explanation coverage,
  top-k composition.
* **Adaptation**: benign FPR before/after online adaptation, memory growth, update/rejection counts,
  alert rate, test metrics after adaptation, `model_weights_updated: false`.
* **Statistics**: seeded percentile bootstrap CIs for F1, AUPRC, ROC-AUC, FPR and Fidelity+.

Design targets from the project documents (FPR < 1.5 %, sub-5 ms latency, ~100k flows/s) are
reported as **targets**; the figure/table code prints them next to measured values and never as
achievements.

## 10. API

```bash
MEMBER3_CONFIG=config.yaml uvicorn member3.api:app --host 0.0.0.0 --port 8040

# explicit paths only - the service auto-discovers and auto-generates nothing:
MEMBER3_RECORDS=/path/to/records.npz MEMBER3_GRAPHS=/path/to/graphs.json \
  MEMBER3_CONFIG=config.yaml uvicorn member3.api:app --port 8040
```

| endpoint | behaviour |
|---|---|
| `GET /` | analyst view (single HTML page served by FastAPI, inline JS, no frontend build) |
| `GET /health` | memory size/capacity, index type, k, threshold + its calibration source, embedding dim, `immutable_memory`, `records_path`, `data_source`, validity warning |
The ASGI app is constructed without touching any artefact (config parse only); the adapter, memory
bank and records resolve on the first request, so `/health` reports the *reason* if something is
missing instead of the server failing to start.

| `POST /score` | accepts `{"embedding":[...]}` or `{"graph":{...}}` or `{"sample_id":"...", "stream_id":"..."}` -> anomaly score, threshold, confidence, neighbours, streak state, alert state |
| `POST /explain` | the same plus the attribution object (top nodes/edges/features, coverage); `explanation_error` when no graph sequence was supplied |
| `GET /metrics` | `results/metrics.json` if present |

The service serves exactly what `config.yaml` points at (no implicit fallback to self-check data):
records for `sample_id` lookups, graph sequences for `/explain`, and `state/` for a reloaded memory
bank. The service is immutable with respect to memory and threshold. `api.allow_online_memory_updates`
(default `false`) is the only switch that lets a scored window enter the bank, and even then only
if it is below threshold and confidently benign. No auth, no database, no queue.

## 11. Figures and tables

Generated by `member3/evaluate.py` from the saved result tables only - deterministic, 300 dpi,
labelled axes with units, legends where needed, no decorative elements. Re-running the same
config on the same records reproduces identical figures; regenerate with
`--stage evaluation` / `--stage xai` / `--stage ablation`.

## 12. Reproducibility

`results/manifest.json` contains: timestamp, git commit (or `unavailable`), dataset identifiers,
split sizes, checkpoint id + SHA-256 (when present), embedding dim, threshold method + value +
scale + calibration source, memory size/version/selection, FAISS configuration, k, neighbour
aggregation, debounce length, adaptation flag, update counts, XAI masking settings, config
SHA-256, the exact command, software versions, hardware info (CPU count, RAM, GPU presence),
per-stage status with failure reasons.

Determinism: one global `seed`; memory selection is farthest-point (k-centre greedy) from a seeded
start; reservoir mode is seeded; bootstrap resampling is seeded; FAISS HNSW construction is greedy.
Repeated runs (`statistics.repeats > 1`) add mean +/- sd.

Verified property (two independent runs of the whole pipeline): `results/paper_tables.csv` is
row-for-row identical **except** the measured wall-clock columns (`TABLE4` latency statistics and
`TABLE9.mean_ms_per_query`), which vary with machine load - by design. Scores, thresholds, memory
state, all classification/zero-day/XAI metrics and every sample count reproduce exactly.
`results/tables.md` and the figures inherit this determinism (tie-stable sorts, no random content).

## 13. Research caveats

1. **Nothing runs without your data.** No dataset, fixture, sample record or stand-in encoder ships
   here, so no number in `results/` can come from anywhere but the upstream artefacts your config
   points at. `pytest` is the only exception and by design: it builds tiny in-memory tensors and a
   test double of the Member-2 *interface* (`tests/test_member3.py::TinyAdapter`) to prove the code
   paths work; those numbers exist nowhere else and are not research results. Running the pipeline
   end to end needs nothing more than `config.yaml` pointed at your artefacts.
2. **No fabricated results.** Unrun experiments are written as `NOT RUN` with the reason (e.g.
   `fig11_cross_dataset_comparison.png` requires a second test dataset). Failed stages stay in the
   manifest.
3. **Leakage controls are code, not prose.** Attack-labelled samples raise `PermissionError` if
   offered to the memory bank; a test `sample_id` in the memory or calibration split, or a
   reference split drawing from a test dataset, aborts the run; evaluation freezes memory and
   threshold (`set_immutable`); `threshold.allow_online_recalibration` is `false` by default.
4. **Attention is Member 2's.** Member 3 ranks, maps and validates attention - it does not invent
   attribution when the adapter exposes none. Feature attribution is labelled
   `derived:attention_weighted_edge_feature_magnitude` (a documented surrogate), never
   "learned feature importance".
5. **Fidelity+ needs re-encoding.** Mask-and-recompute runs through the Member-2 adapter. If no
   masking hook and no maskable sequence format exist, Fidelity+ is `NOT RUN`; embedding-space
   ablation is not reported under the paper's Fidelity+ name.
6. **Delay/episode semantics.** Detection delay is counted in windows of `data.window_seconds`
   (30 s here) and alert-level metrics need `episode_id`; when Member 1 does not supply stream
   ordering, Member 3 derives it and says so in `manifest.provenance.stream_order_source`.
7. **`MODEL ADAPTATION != MEMORY ADAPTATION`.** Online adaptation updates the memory bank only;
   weights are frozen and `model_weights_updated: false` is asserted and recorded.
8. **HNSW is not free.** At small memory sizes exact search is faster than HNSW (see
   `ablations.csv`, arm `retrieval_backend`); the HNSW claim is a scalability claim, and the
   recall-vs-brute-force column quantifies the approximation error.
9. **Threshold percentile is a bound on the calibration split**, not an FPR promise on unseen
   datasets. Cross-dataset FPR is measured on the test split and reported as measured.
10. **Latency is environment-bound.** Report the `conditions` block of `latency.csv` (device,
    k, index type, memory size, CPU count) with any timing figure; do not transfer a timing
    measured here to another machine.

## 14. Layout

```
member3/
├── README.md
├── requirements.txt
├── config.yaml
├── run_experiment.py
├── docs/MEMBER2_HANDOFF_CONTRACT.md   # single upstream integration document (Members 1 + 2)
├── member3/
│   ├── __init__.py
│   ├── contract.py     # upstream interface, records/graph I/O, guards, masking
│   ├── memory.py       # Adaptive Memory Bank + FAISS HNSW
│   ├── inference.py    # score, threshold, alert verification, adaptation, latency, AlertTracker
│   ├── explain.py      # attribution object, Fidelity+, bootstrap
│   ├── evaluate.py     # metrics, ablations, tables, figures, manifest
│   └── api.py          # FastAPI + analyst view
└── tests/test_member3.py

generated by a run, not part of the repository:
├── results/   figures/   state/
└── member2_artifacts/     <- your handoff files (contract §Packaging)
```
 No microservices, no containers, no database, no frontend
framework, no duplicated model code, no mock data of any kind.
