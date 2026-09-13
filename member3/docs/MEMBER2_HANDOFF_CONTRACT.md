# Upstream Handoff Contract - Member 1 + Member 2 -> Member 3

Project: *Adaptive Self-Supervised Temporal Graph Foundation Model for Cross-Dataset Zero-Day Intrusion Detection*

This single file is the complete list of what **Member 1** (data engineering, adaptive windowing,
temporal graph construction) and **Member 2** (self-supervised foundation model) must deliver so
that Member 3 (memory bank, FAISS retrieval, thresholding, alert verification, explainability,
evaluation, API) can run, be reproduced and be defended in review. Forward it as-is.

## 0. Membership map (who must hand over what)

| Deliverable | Owner | Required by Member 3 for | Section |
|---|---|---|---|
| Preprocessed flow records + causality-aware 30 s windows | **Member 1** | proving no look-ahead leakage; window/episode ordering | M1 |
| Temporal graph objects (nodes=hosts, edges=flows, 53 features) | **Member 1** | masking-and-recompute (Fidelity+), node/edge id maps | M1 |
| Split definition (train_benign / calibration_benign / test / online_benign) | **Member 1** | uncontaminated memory, threshold, evaluation | M1.4 |
| `stream_id`, `stream_pos`, `episode_id` per window | **Member 1** | consecutive-anomaly confirmation, detection delay | M1.5 |
| Dataset + preprocessing hashes, feature schema, scaler statistics | **Member 1** (with M2) | reproducibility manifest | M1.6, B.D |
| Model checkpoint + config | **Member 2** | producing embeddings | A |
| `encode()` inference adapter | **Member 2** | embeddings for every window | B |
| `encode_with_attention()` + masking hooks | **Member 2** | GATv2 attribution, Fidelity+ | C |
| Cached embeddings per split (records file) | **Member 2** (export), **Member 1** (metadata columns) | offline evaluation at dataset scale | F |
| Benign/attack + family labels (evaluation only) | **Member 1** export, **Member 2** join | metrics only, never fitted on | F |
| Environment versions, seed, checkpoint checksum | **Member 2** | audit trail per prediction | G |

Member 3 **consumes** these artefacts. It does not download datasets, parse packets, build graphs,
design GATv2/the Temporal Transformer, run NT-Xent or projection-head training, or contain a second
model implementation. If any item below cannot be produced, say so in writing: the corresponding
Member-3 experiment is then reported as `NOT RUN` with the blocking reason, and it is never
estimated.

**There is no fallback in Member 3.** It ships no dataset, no sample records, no fixture encoder and
no demo mode. If an item below is absent or malformed, `run_experiment.py` raises `ContractError`
naming the missing artefact and the section of this document that specifies it. The pipeline is
unit-tested with a small in-memory test double of the interface
(`tests/test_member3.py::TinyAdapter`); that double validates *conformance*, not *performance*, and
it is unreachable from the runner.

Member 3 **consumes** Member 2's artefacts. Member 3 does not reimplement graph construction,
GATv2, the Temporal Transformer, NT-Xent training or the projection head, and it must never be
asked to. If any item below cannot be produced, say so in writing: the corresponding Member-3
experiment is then reported as `NOT RUN` with the blocking reason, and it is never estimated.

Version: `1.0` (schema `SCHEMA_VERSION` in `member3/contract.py`). A change to any shape, dtype
or normalisation rule below is a **breaking change** and must bump this version.

---

## 0. Summary: what Member 3 needs from Member 2

| # | Item | Artefact | Used by Member 3 for |
|---|------|----------|----------------------|
| 1 | Model checkpoint | `member2_artifacts/checkpoint.pt` | producing embeddings |
| 2 | Model config | `member2_artifacts/model_config.yaml` | dimension, architecture id, versions |
| 3 | Inference adapter | `member2_artifacts/model_adapter.py` | `encode()` |
| 4 | Embedding dimension | `metadata()["embedding_dim"]` | memory bank + FAISS index geometry |
| 5 | Embedding normalisation convention | `metadata()["l2_normalized"]`, `["normalization"]` | whether Member 3 must L2-normalise |
| 6 | Exact 53-feature schema | `feature_schema.json` | feature names in explanations |
| 7 | Feature normalisation statistics | `preprocessing.json` | reproducibility of inputs, masking semantics |
| 8 | Graph schema | `preprocessing.json.graph_schema` | node/edge/attr tensors, masking |
| 9 | Temporal sequence schema | `preprocessing.json.sequence_schema` | snapshot ordering |
| 10 | Snapshot ids / timestamps | per-sequence `snapshot_ids`, `timestamps` | detection-latency measurement |
| 11 | Node id -> host/IP map | per-snapshot `node_ids` | human-readable attribution |
| 12 | Edge id -> flow record map | per-snapshot `edge_record_ids` | "which communication caused this" |
| 13 | Benign/attack labels (evaluation only) | `records.npz` column `label` | metrics only, never fitting |
| 14 | Attack-family metadata | `records.npz` column `attack_family` | per-family + leave-one-out zero-day |
| 15 | Attention extraction interface | `encode_with_attention()` | GATv2 structural attribution |
| 16 | Sample input/output example | `sample/` | contract self-test |
| 17 | Environment versions | `metadata()["framework*"]` | manifest, reviewer reproducibility |
| 18 | Random seed | `model_config.yaml: seed` | deterministic embedding reproduction |
| 19 | Checkpoint checksum / version id | `metadata()["checkpoint_version"]` + SHA-256 | auditability of every prediction |

Items 13/14 are **evaluation-only inputs**: labels must never be an input to
representation learning, memory construction or threshold calibration.

---

---

# PART M1 - required from Member 1 (data engineering and graph construction)

## M1.1 Graph object format

Member 3 needs the *same* object Member 2 consumed, addressed by `sample_id`, in one of these two
forms:

1. the portable dict schema in §1 E below (preferred: JSON-serialisable, no torch needed); or
2. Member 1's native PyG object (`Data`/`TemporalData` list) **plus** an adapter function that
   converts it to (1) or exposes the fields Member 3 reads (`x`, `edge_index`, `edge_attr`,
   `node_ids`, `edge_record_ids`). Member 3 will not reimplement your graph classes.

A `--stage xai`/Fidelity+ run needs the actual graphs; cached embeddings alone cannot be masked.

## M1.2 Node and edge identifier maps (mandatory for the XAI contribution)

| Map | Shape | Example | Without it |
|---|---|---|---|
| `node_ids` | `[N_t]` strings per snapshot | `"192.168.10.5"` | alerts cannot name the suspicious host |
| `edge_record_ids` | `[E_t]` strings per snapshot | `"flow-00017"` | alerts cannot name the offending communication |
| index stability | node/edge indices identical in graph, attention and maps | - | every attribution is wrong by construction |

## M1.3 Snapshot / temporal sequence specification

* window duration `W_m` (30 s in this design) and overlap rule;
* flow counters split **proportionally** across intersecting windows (causality-aware slicing) -
  state the exact allocation formula and confirm no flow data is placed before its arrival;
* `T = 5` snapshots per sequence, ordered by ascending `t0`, ids `snapshot_ids` / `graph_ids`;
* window start `timestamp` in seconds (used for detection-delay seconds);
* the burst/episode definition used for stream ordering (see M1.5).

Member 3 uses `W_m` only as a unit label (`data.window_seconds`) and for delay conversion; if your
window differs, change the config value - do not post-scale numbers by hand.

## M1.4 Split definition and non-overlap proof (this is the leakage contract)

Deliver the split assignment, not just the split names:

| split | content rule |
|---|---|
| `train_benign` | NF-CSE-CIC-IDS2018-v3, benign only; **no** window that shares a host/flow with a test window if your preprocessing aggregates across windows |
| `calibration_benign` | benign only, `sample_id`-disjoint from both other splits |
| `test` | NF-UNSW-NB15-v3 (optionally NF-ToN-IoT-v3), benign + attacks |
| `online_benign` | benign drift stream for the adaptation experiment only |

and state, with numbers: split sizes, the assignment rule (time-based? host-based?), and a
statement that no `test` window appears in either reference split. Member 3 enforces id-disjointness
and the benign-only rule in code (`contract.assert_splits_clean`) and aborts on violation - so an
undocumented overlap will surface as a failed run rather than a quiet bias.

## M1.5 Stream / episode ordering (needed for debounce and latency claims)

| column | meaning |
|---|---|
| `stream_id` | one live traffic stream (e.g. `NF-UNSW-NB15-v3:test`) |
| `stream_pos` | integer position in arrival order, monotone, no gaps assumed |
| `episode_id` | contiguous attack burst the window belongs to (`benign` otherwise) |

These make "3 consecutive confirmations" and "detection delay = k windows" measurable. If Member 1
does not emit them, Member 3 derives them with `contract.assign_streams` and records
`provenance.stream_order_source: derived by member3.contract.assign_streams` in the manifest - in
that case the paper must describe the ordering as a Member-3 assumption, not an upstream fact.

## M1.6 Preprocessing metadata

`member2_artifacts/preprocessing.json` (Member 3 documents that path; there is no path-scanning
fallback in the code) with: exact
53-name feature order + `feature_order_sha256`, per-feature scaling statistics and the split they
were fitted on (must be train-only), categorical encodings + vocabulary, missing-value policy,
clipping/transform list, and the file hashes of the raw inputs. Member 3 copies it into
`results/manifest.json`; a reviewer must be able to rebuild the graph tensors from your code plus
this metadata.

---

# PART M2 - required from Member 2 (self-supervised foundation model)

## 1. A. Model checkpoint

Required, exactly:

* `checkpoint.pt` - weights only (or a `state_dict`); no optimiser state, no scheduler, no
  training history.
* `model_config.yaml` - architecture identifier, layer counts, `embedding_dim`, temperature,
  dropout, projection-head config, training seed, dataset identifiers used for training.
* Metadata that the adapter must be able to report at runtime (see §1 B):

| Field | Meaning |
|---|---|
| `architecture_id` | e.g. `gatv2(3)x(mhsa(4 heads)) + proj(256->128)` |
| `checkpoint_version` | short immutable id, e.g. `m2-v3-seed42-step18000` |
| `framework` / `framework_version` | `torch` / version, plus `torch_geometric` version |
| `embedding_dim` | integer, e.g. `128` |
| `device` | `cpu`, `cuda`, or `cpu+cuda` |
| `l2_normalized` | `true` if the returned vector already has unit L2 norm |
| `post_projection_head` | `true` if the embedding comes after the projection head |
| `normalization` | one-line description of the norm applied to the embedding |
| `attention_available` / `masking_available` | booleans for §1 C |
| `trainable` | MUST be `false` for inference artefacts |

Member 3 fails loudly if `embedding_dim` is missing, non-positive, or disagrees with
`config.yaml: member2.embedding_dim`.

## 1. B. Embedding interface

Member 2 exposes a stable, side-effect-free call:

```python
embedding = model.encode(graph_sequence)     # -> np.ndarray or torch.Tensor
```

Contract:

| Property | Requirement |
|---|---|
| input type | one temporal graph sequence object (see §1 E) |
| input shape | `T` snapshots, each with `x [N_t, 53]`, `edge_index [2, E_t]`, `edge_attr [E_t, 53]` |
| output shape | `(D,)` for a single sequence, or `(B, D)` for a batch of `B` sequences |
| `D` | equals `metadata()["embedding_dim"]` |
| dtype | `float32` preferred; `float64` accepted; no `bfloat16`/`fp16` at the interface |
| device | returned on the device reported by `metadata()["device"]`; Member 3 moves it to CPU |
| normalisation | if `l2_normalized: false`, Member 3 applies `z / ||z||` and records `embedding_normalized_by: member3` |
| pre/post projection head | must be declared; the memory bank uses the **post-projection** embedding |
| determinism | same input + same model => bitwise identical output (no dropout at inference, `model.eval()`) |
| statelessness | `encode()` must not mutate the model, the input object, or any global RNG stream used elsewhere |
| batching | optional `encode_batch(list)`; Member 3 falls back to a loop if absent |

Member 3 rejects an adapter whose outputs have a ragged or wrong dimension.

## 1. C. Explanation interface

Required for graph-based attribution:

```python
result = model.encode_with_attention(graph_sequence)
# result = {
#   "embedding":      np.ndarray (D,),            # identical to encode()
#   "node_attention": np.ndarray, shape (N,) or (T, N) or (T, heads, N),
#   "edge_attention": np.ndarray, shape (E_t,) or (T, E_t),
#   "graph_metadata": {
#        "node_ids":          [ [host_0, host_1, ...] per snapshot ]  # original host/IP strings
#        "edge_record_ids":   [ [flow_record_id, ...] per snapshot ]
#        "snapshot_ids":      [ ... ],
#        "timestamps":        [ ... ]  # window start, seconds, matching snapshot ordering
#        "graph_ids":         [ ... ],
#        "feature_names":     [ ...53 names... ],
#        "primary_snapshot":    int,     # optional: snapshot driving the decision
#   }
# }
```

Rules:

1. `node_attention` / `edge_attention` must be **non-negative and sum to ~1 per snapshot**
   (softmax-normalised GATv2 coefficients). Raw logits are not acceptable; Member 3 ranks and
   reports them as importance shares.
2. Attention is aggregated over snapshots and over heads by Member 3 (mean) - so per-snapshot
   or per-head tensors are both fine.
3. `embedding` returned by `encode_with_attention()` must equal `encode()` for the same input;
   Member 3 checks this and fails the run on mismatch.
4. Node/edge **indices** must be stable between the graph object and the attention tensors, and
   must match `node_ids` / `edge_record_ids` positionally. Without that mapping an alert cannot
   be attributed to a host or a flow, and the XAI contribution of the paper is not evidenced.

### If a direct method is impossible

Provide instead (preferred order):

* `model.attention_weights(graph_sequence)` -> same `node_attention` / `edge_attention` keys, or
* forward-hook adapter: `model.attach_attention_hooks()` returning a captured dict, or
* per-node/per-edge scalar scores computed by Member 2 (e.g. attention rollout).

In all cases `graph_metadata` mapping is still mandatory. If none is available, Member 3 marks
Fidelity+ and attribution `NOT RUN` with reason `attention extraction unavailable` - it does not
substitute a heuristic and report it as the paper's explanation.

### Masking hooks (needed for Fidelity+)

Fidelity+ requires **recomputing the score after removing a component**, so Member 2 must supply
one of:

* `model.mask_nodes(graph_sequence, node_ids) -> masked_sequence` and
  `model.mask_edges(graph_sequence, edge_ids) -> masked_sequence`, or
* a documented sequence format Member 3 can mask directly (then `member3.contract.mask_graph_sequence`
  is used: node removal drops the node and its incident edges; edge removal zeroes `edge_attr` rows).

If neither is possible, the node-masking Fidelity+ experiment cannot be measured.

## 1. D. Feature schema (`feature_schema.json`)

```json
{
  "schema_version": "1.0",
  "source": "NF-CSE-CIC-IDS2018-v3 NetFlow v3",
  "n_features": 53,
  "feature_names": ["dur", "proto", "service", "state", "spackets", "... 53 total ..."],
  "feature_order_sha256": "<sha256 of the newline-joined ordered names>",
  "dtype": "float32",
  "categorical": [{"name": "proto", "encoding": "ordinal", "vocabulary": ["tcp", "udp", "icmp"]}],
  "missing_value_policy": "indicator column + median imputation",
  "scaling": {"method": "standard", "mean": ["...53..."], "std": ["...53..."]},
  "clip": null,
  "fit_scope": "train split of NF-CSE-CIC-IDS2018-v3 only"
}
```

Requirements:

* the **exact 53-name order** the model consumes (a reordered vector silently corrupts every
  number in the paper);
* scaler statistics + the split they were fitted on (no test-split statistics);
* categorical encoding rules and vocabulary; missing-value policy; any clipping/transform.

Member 3 compares `n_features` and the name list against what `graph_metadata["feature_names"]`
returns and fails on mismatch.

## 1. E. Graph schema

Expected sequence object (dict is the portable representation; a PyG
`[Data]`/`[DynamicData]` object is fine if the adapter accepts it and exposes the same fields):

```python
graph_sequence = {
  "sample_id":   "tes-000100",           # window/sequence id, unique across splits
  "dataset_id":  "NF-UNSW-NB15-v3",
  "window_id":   "tes-w000100",
  "n_snapshots": 5,
  "snapshot_ids": ["tes-000100-t500", "..."],
  "timestamps":   [15000.0, 15030.0, "..."],   # window start seconds, W_m = 30 s
  "graph_ids":    ["tes-000100-g0", "..."],
  "feature_names": ["dur", "... 53 ..."],
  "snapshots": [
     {
      "num_nodes": 7,
      "x":        [[...53...], ...],     # [N, 53] node features (aggregate of its flows)
      "edge_index": [[...srcs...], [...dsts...]],   # [2, E] int64, indices into x
      "edge_attr": [[...53...], ...],    # [E, 53] NetFlow v3 features per communication
      "node_ids":        ["192.168.10.5", "..."],   # index -> host/IP
      "edge_record_ids": ["flow-00017", "..."],     # index -> original flow record
      "t0": 15000.0, "t1": 15030.0
     }, ...
  ]
}
```

| Aspect | Requirement |
|---|---|
| node representation | one node per host/IP active in the snapshot; `x` = aggregated flow features |
| edge representation | directed communication; `edge_index` column order defines edge id |
| edge attributes | the 53 NetFlow v3 features, already scaled |
| node attributes | aggregation of incident edges (state the aggregation rule) |
| sequence | fixed `T = 5` snapshots ordered by ascending `t0` |
| snapshot boundaries | causality-aware 30 s windows; flow counters split proportionally over intersecting windows |
| graph ids | unique, stable, joinable to Member 1's records |
| node/edge index stability | indices within a snapshot are never re-sorted between `encode` and `encode_with_attention` |

## 1. F. Label / attack metadata (evaluation only)

Member 3 reads one **records file** (`.npz` preferred, `.csv` accepted):

| column | type | notes |
|---|---|---|
| `dataset_id` | str | e.g. `NF-UNSW-NB15-v3` |
| `split` | str | `train_benign` \| `calibration_benign` \| `test` \| `online_benign` |
| `sample_id` | str | unique |
| `window_id` | str | Member 1's window id |
| `timestamp` | float | window start, seconds |
| `embedding` | float32 `[D]` | output of `encode()` for that sequence |
| `label` | str | `benign` \| `attack` |
| `attack_family` | str | family name, `benign` for normal traffic |

Split semantics (this is the leakage contract, enforced by code in `member3/contract.py:assert_splits_clean`):

| split | content | Member 3 use |
|---|---|---|
| `train_benign` | NF-CSE-CIC-IDS2018-v3, **benign only** | builds the Adaptive Memory Bank |
| `calibration_benign` | benign only, disjoint from `train_benign`, disjoint from test | threshold calibration **only** |
| `test` | NF-UNSW-NB15-v3 (optionally + NF-ToN-IoT-v3), benign + attacks | metrics only; never fits anything |
| `online_benign` | benign drift stream, labelled as such | online memory-adaptation experiment only |

Additionally required for the temporal/operational metrics: `stream_id`, `stream_pos`
(monotone ordering within a live stream). If Member 1/2 do not supply them, Member 3 assigns them
with `contract.assign_streams`, and the manifest records that ordering was Member-3-derived.

Preferably ship cached embeddings for the full corpus: the paper run then needs no GPU in Member 3,
and Member-3 timings stay separable from Member-2 encode time. Also ship the graph sequences (or
the .pt files plus the adapter) for the windows chosen for Fidelity+; without them masking-and-recompute
cannot be measured.

Attack labels MUST NOT be inputs to representation learning, memory construction or threshold
calibration. If Member 2's training used any attack data, that must be stated here and in the
paper; the "normal-only" claim would otherwise be unsupported.

## 1. G. Reproducibility

Provide: `seed`, full model config, training checkpoint metadata (steps, wall-clock, hardware),
software versions (python, torch, torch_geometric, CUDA), preprocessing code version or
`preprocessing.json` hash, dataset file hashes, and the split definition (which windows, by what
rule). Member 3 copies all of it into `results/manifest.json`.

## 1. H. Artifacts (exactly these, nothing more)

```
member2_artifacts/
├── checkpoint.pt
├── model_config.yaml
├── preprocessing.json          # graph + sequence schema, scaler stats, windowing params
├── feature_schema.json         # exact 53-feature order + encoding rules
├── model_adapter.py            # FoundationModelAdapter (see §2)
└── sample/
    ├── sample_graph.npz        # one real sequence (or .pt if torch-only)
    ├── sample_embedding.npy    # encode(sample_graph)          [D]
    ├── sample_attention.npz    # encode_with_attention(sample_graph) verbatim
    └── sample_record.json      # the matching row of records.npz
```

No training logs, no optimiser state, no dataset copies, no notebooks. The `sample/` bundle is the
self-test sample: Member 3 loads it and asserts shape, dtype, unit norm, determinism,
embedding/attention index consistency, and that `encode_with_attention()["embedding"]` equals
`encode()`.

---

## 2. Expected Python interface

```python
# model_adapter.py  (Member 2 owns this file; Member 3 imports it)
class FoundationModelAdapter:
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_config_path: str) -> "FoundationModelAdapter": ...
    def metadata(self) -> dict: ...                       # keys from §1 A (mandatory)
    def encode(self, graph_sequence) -> "np.ndarray": ...  # (D,) or (B, D)
    def encode_with_attention(self, graph_sequence) -> dict: ...   # §1 C
    # optional, needed for node/edge-masking Fidelity+:
    def mask_nodes(self, graph_sequence, node_ids) -> "graph_sequence": ...
    def mask_edges(self, graph_sequence, edge_ids) -> "graph_sequence": ...
    def encode_batch(self, sequences) -> "np.ndarray": ...
```

Load path used by Member 3 (`member3/contract.py:load_adapter`):

```python
from model_adapter import FoundationModelAdapter
model = FoundationModelAdapter.from_checkpoint("member2_artifacts/checkpoint.pt",
                                               "member2_artifacts/model_config.yaml")
embedding = model.encode(graph_sequence)
explanation = model.encode_with_attention(graph_sequence)
```

`model_config.yaml` may be any format the adapter understands; Member 3 only needs the file to
exist and the adapter to report metadata. A runnable reference implementation of the whole
interface - used by Member 3's own contract tests - exists in
`member3/contract.py:SelfCheckFoundationAdapter`. It is a deterministic NumPy stand-in (random
projection over snapshot statistics) that must **never** be presented as the research encoder; it
exists only so the Member-3 pipeline can be validated before the real artefacts arrive, and every
result can come from it.

## 3. What Member 2 must implement

1. A frozen inference wrapper exposing `metadata`, `encode`, `encode_with_attention`.
2. `model.eval()` + `torch.no_grad()` semantics so outputs are deterministic.
3. Attention extraction from the GATv2 layers, softmax-normalised, with the node/edge index maps.
4. Masking hooks (`mask_nodes` / `mask_edges`) or a documented sequence format Member 3 can mask.
5. Cached-embedding export for all four splits into the records format of §1 F.
6. `feature_schema.json` + `preprocessing.json` (exact 53-feature order, scaler statistics).
7. The `sample/` self-test bundle.
8. A statement of the training data policy (benign only) with the split hash used.

## 4. What Member 2 must NOT implement

* no memory bank, FAISS index, thresholding, debounce, metrics, figures, tables, API, dashboard;
* no retraining/fine-tuning hooks exposed to Member 3 (`trainable` must be `false`);
* no label leakage into embeddings: no supervised head, no attack-aware normalisation;
* no dataset downloading/parsing/preprocessing inside the adapter (Member 1 owns it);
* no duplicated copies of Member 1 graph objects to "make life easier" - document the format instead;
* no silent normalisation changes between runs (any change must bump `schema_version`).

## 5. Packaging

* one directory, `member2_artifacts/`, self-contained; a `.tar.gz` + its SHA-256 is enough;
* no absolute paths inside any file; the adapter must resolve `checkpoint_path` relative to
  the artefact dir it was given;
* single-source dependencies: if the adapter imports `torch`, ship the exact versions in
  `model_config.yaml: environment`;
* do **not** ship a second model implementation for Member 3 to "pick"; one adapter, one path.

## 6. Expected tensor shapes (quick table)

| Object | Shape | dtype |
|---|---|---|
| `snapshots[t]["x"]` | `[N_t, 53]` | float32 |
| `snapshots[t]["edge_index"]` | `[2, E_t]` | int64 |
| `snapshots[t]["edge_attr"]` | `[E_t, 53]` | float32 |
| sequence | `T = 5` snapshots | - |
| `encode` output | `[D]`, e.g. `[128]` | float32 |
| `encode` batch output | `[B, D]` | float32 |
| `node_attention` | `[N]` or `[T, N]` or `[T, heads, N]` | float32 |
| `edge_attention` | `[E_t]` or `[T, E_t]` | float32 |
| records `embedding` column | `[N_samples, D]` | float32 |

## 7. Example calls (what Member 3 will actually run)

Member 3 resolves the adapter in `contract.load_adapter`: the file at `member2.adapter_path`, or
`<member2.artifacts_dir>/model_adapter.py`. If neither exists the run aborts - there is no second
code path that could quietly substitute something else.

```python
adapter, meta = load_adapter(cfg)              # metadata validated, dim checked
X = np.vstack(list(records["embedding"]))      # cached path (preferred for large corpora)
scorer = Member3Scorer.build(cfg, adapter, meta, records)   # memory + threshold, benign splits only
scorer.memory.set_immutable(True)               # evaluation cannot move the reference set
pred = scorer.verify_alerts(scorer.predict(records))
explanation = build_explanation(adapter, graphs[sample_id], score, threshold, confidence, alert_id)
fid_df, fid_summary = fidelity_plus(scorer, alert_rows, graphs, cfg)   # mask -> re-encode -> re-score
```

## 8. Validation checklist

Member 2 sends this back ticked; Member 3's `--stage calibration` re-checks items 1-9 automatically.

- [ ] 1 `checkpoint.pt` loads and `metadata()["embedding_dim"]` == config `member2.embedding_dim`
- [ ] 2 `encode()` returns shape `(D,)` / `(B, D)`, dtype float32/float64, finite everywhere
- [ ] 3 `encode()` is deterministic across 3 calls (bitwise) on the same input
- [ ] 4 embeddings are unit-norm, or `l2_normalized: false` is declared
- [ ] 5 `encode_with_attention()["embedding"]` == `encode()` for the same input
- [ ] 6 attention tensors are non-negative, sum to 1 per snapshot, and match node/edge index ranges
- [ ] 7 `node_ids` / `edge_record_ids` are present, positional, and resolve to real hosts/flows
- [ ] 8 masking hook exists, or the sequence format is maskable by `member3.contract`
- [ ] 9 records file has all 8 mandatory columns, 4 split tags, benign-only in the two reference splits
- [ ] 10 `feature_schema.json` lists exactly 53 names in model order, with `feature_order_sha256`
- [ ] 11 scaler statistics state the fit split (train-only)
- [ ] 12 `sample/` bundle reproduces byte-identical outputs on the reviewer machine
- [ ] 13 training data policy (benign-only) stated in writing
- [ ] 14 software versions + seed + checkpoint checksum supplied
- [ ] 15 any deviation from this contract is listed explicitly (never silently)

### Member 1 checklist

- [ ] M1 graph objects addressable by `sample_id`, in the portable dict schema or behind a converter
- [ ] M2 `node_ids` / `edge_record_ids` present, positional, resolving to real hosts/flows
- [ ] M3 snapshot ordering + `W_m` + proportional overlap allocation formula documented, no look-ahead
- [ ] M4 four split tags with sizes and an explicit non-overlap statement
- [ ] M5 `stream_id` / `stream_pos` / `episode_id` exported (or Member-3 derivation accepted in writing)
- [ ] M6 `preprocessing.json` with 53-name order, scaler statistics fitted on train only, input hashes
- [ ] M7 labels/families carried into the records file without touching any representation-learning input
- [ ] M8 the exact command/script that regenerates the graph tensors from the raw inputs

## 9. Failure semantics (agreed)

Member 3 aborts - rather than continuing with a weakened protocol - when the checkpoint is
missing, `embedding_dim`/feature/graph schemas mismatch, calibration data is absent or too small,
evaluation labels are missing, the FAISS index dimension disagrees, attention extraction is
unavailable, or a split is contaminated. Aborted experiments are recorded in
`results/manifest.json` with their reason; they are never replaced by estimated numbers.
