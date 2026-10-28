"""Smoke-test the real Member-2 checkpoint through the Member-3 adapter bridge.

Run from the member3 project directory:
    python tools/smoke_test_member2_adapter.py --pairs member2_artifacts/sample_pairs.pt

Requires the local checkpoint/config and the copied Member-2 sample_pairs.pt. It does not retrain.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow execution as `python tools/smoke_test_member2_adapter.py` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from member3.contract import load_config, load_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pairs", default="member2_artifacts/sample_pairs.pt")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    adapter, meta = load_adapter(cfg)
    pairs_path = Path(args.pairs)
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Sample pairs missing: {pairs_path}. Copy sample_pairs.pt from Member 2's package.")
    pairs = torch.load(pairs_path, map_location="cpu", weights_only=False)
    if not isinstance(pairs, (list, tuple)) or len(pairs) < 2:
        raise ValueError("sample_pairs.pt must contain at least one benign and one attack graph sequence")

    worst = 0.0
    for pair in pairs:
        snapshots = pair["snapshots"]
        expected = pair.get("h")
        actual = np.asarray(adapter.encode(snapshots), dtype=np.float32).reshape(-1)
        if actual.shape != (128,):
            raise AssertionError(f"encode() returned {actual.shape}, expected (128,)")
        norm = float(np.linalg.norm(actual))
        if not np.isclose(norm, 1.0, atol=1e-4):
            raise AssertionError(f"h is not L2-normalised: norm={norm}")
        if expected is not None:
            expected_np = expected.detach().cpu().numpy() if torch.is_tensor(expected) else np.asarray(expected)
            diff = float(np.max(np.abs(actual - expected_np.reshape(-1))))
            worst = max(worst, diff)
            print(f"sample_id={pair.get('source_index', '?')} label={pair.get('label', '?')} max_abs_diff={diff:.3e} norm={norm:.6f}")
            if diff > args.tolerance:
                raise AssertionError(f"cached h mismatch: max absolute difference {diff:.3e} > {args.tolerance}")
        attention = adapter.encode_with_attention(snapshots)
        for key in ("embedding", "node_attention", "edge_attention", "graph_metadata"):
            if key not in attention:
                raise AssertionError(f"encode_with_attention() is missing '{key}'")
        attention_diff = float(np.max(np.abs(np.asarray(attention["embedding"]) - actual)))
        if attention_diff > max(args.tolerance, 1e-3):
            raise AssertionError(f"attention embedding doesn't match encode(): {attention_diff:.3e}")
    print("PASS: checkpoint, adapter contract, 128-D h, unit norm and attention path verified")
    print(f"checkpoint_version={meta.get('checkpoint_version')} embedding_dim={meta.get('embedding_dim')} worst_cached_diff={worst:.3e}")


if __name__ == "__main__":
    main()
