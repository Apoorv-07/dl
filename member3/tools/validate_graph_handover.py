#!/usr/bin/env python3
"""Audit Member-1 graph exports against Member-2's embedding-record IDs.

Loads one large .pt file at a time. This still needs enough RAM to deserialize a single graph
file; it deliberately does not retain both multi-GB graph objects at the same time.
It does not alter graph files or make final evaluation claims.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from member3.contract import load_records  # noqa: E402


def getv(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def graph_dataset_id(path: str, blob: dict) -> str:
    name = Path(path).name.lower()
    if "graph_validation_v2" in name:
        return "dataset2_graph_validation_v2"
    if name == "graph.pt" or name.startswith("graph_train"):
        return "dataset1_graph_pt"
    return str(blob.get("dataset", ""))


def audit_graph(path: str, records: pd.DataFrame, expected_features: int, expected_snapshots: int) -> dict:
    import torch

    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    print(f"[audit] loading one graph file: {path}", flush=True)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or not isinstance(blob.get("sequences"), (list, tuple)):
        raise ValueError(f"{path}: expected dict containing a sequences list")
    dsid = graph_dataset_id(path, blob)
    sequences = blob["sequences"]
    ids = []
    snapshots_per_sequence = Counter()
    node_dims = Counter()
    edge_dims = Counter()
    snapshots_with_time = 0
    snapshot_count = 0
    bad_sequence_ids = 0
    bad_shapes = []
    empty_edges = 0
    labels = Counter()

    for i, seq in enumerate(sequences):
        sid = getv(seq, "sequence_id", getv(seq, "sample_id", None))
        if sid is None:
            bad_sequence_ids += 1
            sid = f"__missing_sequence_id_at_index_{i}"
        sid = str(sid)
        ids.append(sid)
        snaps = getv(seq, "snapshots", []) or []
        snapshots_per_sequence[len(snaps)] += 1
        lab = getv(seq, "label", getv(seq, "y", None))
        if lab is not None:
            labels[str(lab)] += 1
        for j, snap in enumerate(snaps):
            snapshot_count += 1
            x = getv(snap, "x")
            edge_index = getv(snap, "edge_index")
            edge_attr = getv(snap, "edge_attr")
            shape_x = tuple(getattr(x, "shape", ()))
            shape_ei = tuple(getattr(edge_index, "shape", ()))
            shape_ea = tuple(getattr(edge_attr, "shape", ()))
            node_dims[shape_x[-1] if shape_x else None] += 1
            edge_dims[shape_ea[-1] if shape_ea else None] += 1
            if shape_ei == (2, 0) or (len(shape_ei) == 2 and shape_ei[-1] == 0):
                empty_edges += 1
            if shape_x and shape_x[-1] != expected_features:
                bad_shapes.append({"sequence_id": sid, "snapshot_index": j, "field": "x", "shape": shape_x})
            if shape_ea and shape_ea[-1] != expected_features:
                bad_shapes.append({"sequence_id": sid, "snapshot_index": j, "field": "edge_attr", "shape": shape_ea})
            if shape_ei and (len(shape_ei) != 2 or shape_ei[0] != 2):
                bad_shapes.append({"sequence_id": sid, "snapshot_index": j, "field": "edge_index", "shape": shape_ei})
            if any(getv(snap, k, None) is not None for k in ("timestamp", "start_time", "end_time", "window_id")):
                snapshots_with_time += 1

    duplicate_ids = sorted(k for k, n in Counter(ids).items() if n > 1)
    graph_ids = set(ids)
    rec = records.loc[records["dataset_id"].astype(str).eq(dsid)].copy()
    record_ids = set(rec["sample_id"].astype(str)) if len(rec) else set()
    missing_from_graph = sorted(record_ids - graph_ids)
    absent_from_records = sorted(graph_ids - record_ids)
    duplicate_record_ids = sorted(rec.loc[rec["sample_id"].astype(str).duplicated(keep=False), "sample_id"].astype(str).unique()) if len(rec) else []
    summary = {
        "path": str(Path(path).resolve()),
        "dataset_id_assumed_from_filename": dsid,
        "file_bytes": os.path.getsize(path),
        "sequence_count": len(sequences),
        "snapshot_count": snapshot_count,
        "snapshots_per_sequence_distribution": {str(k): int(v) for k, v in sorted(snapshots_per_sequence.items())},
        "expected_snapshots_per_sequence": expected_snapshots,
        "node_feature_dimensions": {str(k): int(v) for k, v in node_dims.items()},
        "edge_feature_dimensions": {str(k): int(v) for k, v in edge_dims.items()},
        "expected_feature_dimension": expected_features,
        "snapshots_with_explicit_time_metadata": snapshots_with_time,
        "sequence_ids_missing": bad_sequence_ids,
        "duplicate_sequence_ids_count": len(duplicate_ids),
        "duplicate_sequence_ids_sample": duplicate_ids[:25],
        "sequence_labels_if_present": dict(labels),
        "snapshots_with_empty_edge_index": empty_edges,
        "shape_errors_count": len(bad_shapes),
        "shape_errors_sample": bad_shapes[:25],
        "records_for_dataset_count": len(rec),
        "records_unique_sample_ids_count": len(record_ids),
        "record_ids_not_found_in_graph_count": len(missing_from_graph),
        "record_ids_not_found_in_graph_sample": missing_from_graph[:25],
        "graph_ids_not_found_in_records_count": len(absent_from_records),
        "graph_ids_not_found_in_records_sample": absent_from_records[:25],
        "duplicate_record_sample_ids_count": len(duplicate_record_ids),
        "duplicate_record_sample_ids_sample": duplicate_record_ids[:25],
    }
    del sequences, blob
    gc.collect()
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True, help="Path to the benign training graph .pt")
    p.add_argument("--validation", required=True, help="Path to mixed validation graph .pt")
    p.add_argument("--records", default="member2_artifacts/member2_records.npz")
    p.add_argument("--out", default="results/graph_handover_audit.json")
    p.add_argument("--features", type=int, default=53)
    p.add_argument("--snapshots", type=int, default=5)
    args = p.parse_args()

    records = load_records(args.records)
    summaries = []
    for path in (args.train, args.validation):
        summaries.append(audit_graph(path, records, args.features, args.snapshots))
    result = {
        "status": "AUDIT ONLY — not final evaluation",
        "records_path": str(Path(args.records).resolve()),
        "record_rows": int(len(records)),
        "record_counts_by_dataset_split_label": {
            "|".join(map(str, key)): int(n)
            for key, n in records.groupby(["dataset_id", "split", "label"], dropna=False).size().items()
        },
        "graph_files": summaries,
        "checks_to_resolve_before_final_metrics": [
            "Confirm every graph sequence_id maps to the intended Member-2 sample_id.",
            "Resolve all record/graph ID/count mismatches; do not map by list index unless Member 1 confirms that contract.",
            "Verify timestamp restoration happened before 30-second window construction.",
            "Verify duplicate snapshots against source-flow IDs and split boundaries before declaring leakage.",
            "Keep labels out of prediction, calibration fitting, memory updates, and XAI candidate selection.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[audit] wrote {out}")
    for g in summaries:
        print(f"\n{g['path']}")
        print(f"  sequences={g['sequence_count']} snapshots={g['snapshot_count']} "
              f"snapshots/sequence={g['snapshots_per_sequence_distribution']}")
        print(f"  node_dims={g['node_feature_dimensions']} edge_dims={g['edge_feature_dimensions']}")
        print(f"  record ids missing from graph={g['record_ids_not_found_in_graph_count']} "
              f"graph ids missing from records={g['graph_ids_not_found_in_records_count']} "
              f"duplicate graph IDs={g['duplicate_sequence_ids_count']}")
        print(f"  explicit snapshot time metadata={g['snapshots_with_explicit_time_metadata']} "
              f"shape errors={g['shape_errors_count']}")
    print("\nAudit only: do not report final performance until split, timestamp, and overlap checks are cleared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
