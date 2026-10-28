# Member 1 → Member 3 graph handover: audit findings and next run

## Configuration update

`config.yaml` now points `data.graph_path` to `../member2/graph_validation_v2.pt`. The training graph is deliberately excluded from the XAI process: it is not required for explanations of validation/test records and it has a four-record mismatch against the packaged record table.

Run commands from the `dl/member3` directory. Do not commit either multi-GB `.pt` file to Git.

## Audit findings supplied on 2026-10-28

- Validation graph: 2,463 sequences, 12,315 snapshots, five snapshots per sequence, and 53 node/edge features per snapshot.
- Validation graph ID comparison: 2,463 record IDs, no unmatched record IDs and no graph IDs absent from records. This is a **key/count match**, not independent proof that `sequence_id` semantically identifies the same sequence as Member 2's `sample_id`; Member 1 must confirm that contract.
- Training graph: 2,463 sequences, while record metadata has 2,467 rows for `dataset1_graph_pt`. IDs `2463`–`2466` have no matching graph sequence. Do not infer a mapping by list order or silently drop these records.
- Both graphs report zero snapshots with explicit time metadata. Verify that timestamp overflow was corrected before the 30-second graph windowing procedure.
- Exact graph-snapshot duplicates were observed upstream. Exact duplicates alone do not prove source-flow leakage; verify shared source-flow IDs and split boundaries.
- Cached-embedding readout testing reported best ROC-AUC 0.463, below random ranking. Detector quality remains unestablished and should not be reported as a successful zero-day result.

## Run sequence

```powershell
python -m pytest -q
python tools\smoke_test_member2_adapter.py --pairs member2_artifacts\sample_pairs.pt
python toolsalidate_graph_handover.py --train ..\member2\graph.pt --validation ..\member2\graph_validation_v2.pt --records member2_artifacts\member2_records.npz
python run_experiment.py --config config.yaml --fail-fast
```

The audit command reads the files one at a time but still requires enough RAM to deserialize each file. The experiment runner then loads the validation graph for evaluation's graph benchmark and XAI. It does not load the training graph.

## Expected XAI outputs

- `results/explanations.json`: explanations for predicted test anomalies whose IDs match validation graph sequences.
- `results/fidelity_plus.csv`: masking-based Fidelity+ rows if the model, graph match and masking evaluation all succeed.
- `results/manifest.json`: stage status, matching/missing graph counts and any NOT RUN reasons.
- `figures/figure_status.json`: which figures were generated or skipped.

The current prediction output flagged only one test record, a benign false positive, and zero confirmed alerts. Therefore, successful XAI may explain that false positive; it will not mean the detector has started detecting attacks. Graphs being loaded also does not resolve timestamp alignment, possible leakage or the poor ROC-AUC.
