"""Compatibility bridge from Member 2's native adapter to Member 3's interface.

The real encoder/checkpoint remain owned by Member 2. This file only adapts naming, input graph
format, metadata and attention shape to Member 3; it does not implement a second model.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_native_module():
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # <repo>/member3/member2_artifacts/model_adapter.py -> <repo>
    default_path = repo_root / "member2" / "training" / "model_adapter.py"
    source = Path(os.environ.get("MEMBER2_ADAPTER_SOURCE", str(default_path))).resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Member-2's real adapter was not found at {source}. Keep this repository layout "
            "(member2/training/model_adapter.py next to member3/) or set MEMBER2_ADAPTER_SOURCE."
        )
    training_dir = str(source.parent)
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)
    spec = importlib.util.spec_from_file_location("member2_native_adapter_for_member3", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Member-2 adapter at {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, source


def _to_numpy(value: Any, dtype=None) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def _ordered_ids(value: Any, expected_n: int) -> List[str]:
    """Convert the native node-id map to node-index order when its structure is identifiable."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        # Common native form: a list of IPs indexed in the same order as graph.x.
        if value and isinstance(value[0], (list, tuple, dict)):
            return _ordered_ids(value[0], expected_n)
        return [str(v) for v in value[:expected_n]]
    if isinstance(value, dict):
        keys = list(value.keys())
        if all(str(k).isdigit() for k in keys):
            keys.sort(key=lambda k: int(str(k)))
            return [str(value[k]) for k in keys[:expected_n]]
        # Reverse an IP -> integer-index map only when values are all integer-like.
        try:
            items = sorted(value.items(), key=lambda kv: int(kv[1]))
            if all(int(v) >= 0 for _, v in items):
                return [str(k) for k, _ in items[:expected_n]]
        except (TypeError, ValueError):
            pass
    return []


class FoundationModelAdapter:
    """Member-3 contract wrapper around Member2Adapter."""

    def __init__(self, native_adapter, checkpoint_path: str, model_config_path: str, native_source: Path):
        self._native = native_adapter
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.model_config_path = str(Path(model_config_path).resolve())
        self.native_source = str(native_source)
        self.embedding_dim = 128

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_config_path: str) -> "FoundationModelAdapter":
        ckpt = Path(checkpoint_path).resolve()
        config = Path(model_config_path).resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Member-2 checkpoint missing: {ckpt}")
        if not config.is_file():
            raise FileNotFoundError(f"Member-2 model config missing: {config}")
        try:
            model_config = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            raise ValueError(f"Cannot parse Member-2 model config {config}: {exc}") from exc
        expected_hash = (model_config.get("model", {}) or {}).get("checkpoint_sha256") or model_config.get("checkpoint_sha256")
        actual_hash = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        if expected_hash and str(expected_hash).lower() != actual_hash.lower():
            raise ValueError(
                f"Checkpoint SHA-256 mismatch: config expects {expected_hash}, but {ckpt.name} is {actual_hash}. "
                "Copy the checkpoint that belongs to this model_config.yaml."
            )
        module, source = _load_native_module()
        if not hasattr(module, "Member2Adapter"):
            raise AttributeError(f"{source} must expose Member2Adapter")
        native = module.Member2Adapter(str(ckpt))  # strict checkpoint load + eval mode in Member 2
        return cls(native, str(ckpt), str(config), source)

    def metadata(self) -> Dict[str, Any]:
        raw = dict(self._native.metadata() or {})
        h = hashlib.sha256(Path(self.checkpoint_path).read_bytes()).hexdigest()
        versions = raw.get("versions", {}) or {}
        return {
            **raw,
            "architecture_id": "gatv2-temporal-transformer-h128",
            "checkpoint_version": h[:16],
            "checkpoint_sha256": h,
            "framework": "torch",
            "framework_version": str(versions.get("torch", "unknown")),
            "embedding_dim": 128,
            "l2_normalized": True,
            "post_projection_head": False,
            "normalization": "Member 2's h: temporal mean -> LayerNorm -> L2 normalize; z projection is not scored",
            "attention_available": True,
            "masking_available": True,
            "trainable": False,
            "device": "cuda" if "cuda" in str(versions.get("device", "" )).lower() else "runtime-selected",
            "adapter_bridge": "member3.member2_artifacts.model_adapter.FoundationModelAdapter",
            "native_adapter_source": self.native_source,
        }

    def _snapshots(self, graph_sequence: Any):
        """Accept Member-2 PyG Data lists or Member-3's portable dict sequence."""
        if isinstance(graph_sequence, dict):
            raw_snapshots = graph_sequence.get("snapshots")
            if raw_snapshots is None:
                raise ValueError("graph sequence dict must contain 'snapshots'")
        else:
            raw_snapshots = graph_sequence
        if not isinstance(raw_snapshots, (list, tuple)):
            raise TypeError("graph_sequence must be a dict with snapshots or a list of PyG Data")
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as exc:
            raise RuntimeError("Member-2 graph inference requires torch and torch_geometric") from exc
        out = []
        for i, snapshot in enumerate(raw_snapshots):
            if hasattr(snapshot, "x") and hasattr(snapshot, "edge_index"):
                # Avoid mutating the original dataset objects.
                out.append(snapshot.clone())
                continue
            if not isinstance(snapshot, dict):
                raise TypeError(f"snapshot {i} must be a PyG Data object or dict")
            x = _to_numpy(snapshot.get("x"), np.float32)
            ei = _to_numpy(snapshot.get("edge_index"), np.int64).reshape(2, -1)
            ea = _to_numpy(snapshot.get("edge_attr"), np.float32)
            if x.ndim != 2 or x.shape[1] != 53:
                raise ValueError(f"snapshot {i}: node x must have shape [N,53], got {x.shape}")
            if ea.size and (ea.ndim != 2 or ea.shape[1] != 53):
                raise ValueError(f"snapshot {i}: edge_attr must have shape [E,53], got {ea.shape}")
            data = Data(x=torch.as_tensor(x, dtype=torch.float32),
                        edge_index=torch.as_tensor(ei, dtype=torch.long),
                        edge_attr=torch.as_tensor(ea, dtype=torch.float32))
            out.append(data)
        if len(out) != 5:
            raise ValueError(f"Member 2 expects 5 snapshots per sequence; received {len(out)}")
        return out

    def encode(self, graph_sequence) -> np.ndarray:
        snapshots = self._snapshots(graph_sequence)
        vec = np.asarray(self._native.encode(snapshots), dtype=np.float32).reshape(-1)
        if vec.shape != (128,):
            raise ValueError(f"Member-2 encode returned shape {vec.shape}; expected (128,)")
        if not np.all(np.isfinite(vec)):
            raise ValueError("Member-2 embedding contains NaN/Inf")
        return vec

    def encode_with_attention(self, graph_sequence) -> Dict[str, Any]:
        snapshots = self._snapshots(graph_sequence)
        raw = self._native.encode_with_attention(snapshots)
        h = np.asarray(raw["h"], dtype=np.float32).reshape(-1)
        if h.shape != (128,):
            raise ValueError(f"Member-2 attention path returned h shape {h.shape}; expected (128,)")
        if not isinstance(graph_sequence, dict):
            seq_meta: Dict[str, Any] = {}
            portable_snapshots = snapshots
        else:
            seq_meta = graph_sequence
            portable_snapshots = graph_sequence["snapshots"]

        # Member 2 returns edge attention from GATv2. Derive node importance by accumulating
        # first-snapshot incident attention, and disclose this as an aggregation in metadata.
        first = portable_snapshots[0]
        if isinstance(first, dict):
            x = _to_numpy(first.get("x"), np.float32)
            edge_index = _to_numpy(first.get("edge_index"), np.int64).reshape(2, -1)
        else:
            x = _to_numpy(first.x, np.float32)
            edge_index = _to_numpy(first.edge_index, np.int64).reshape(2, -1)
        n_nodes = int(x.shape[0]) if x.ndim >= 1 else 0
        n_edges = int(edge_index.shape[1])
        node_attention = np.zeros(n_nodes, dtype=np.float64)
        edge_attention = np.zeros(n_edges, dtype=np.float64)
        gat_items = raw.get("gat_attention", []) or []
        if gat_items and n_nodes:
            entry = gat_items[0].get("layer1", None)
            if entry is not None:
                attn_edge_index = _to_numpy(entry[0], np.int64).reshape(2, -1)
                alpha = _to_numpy(entry[1], np.float64)
                if alpha.ndim == 2:
                    alpha = alpha.mean(axis=1)
                alpha = alpha.reshape(-1)
                # PyG appends one self-loop per node. Original edges are the leading E entries
                # for this trained model's configuration; cap defensively for unusual graphs.
                usable = min(n_edges, len(alpha), attn_edge_index.shape[1])
                if usable:
                    edge_attention[:usable] = alpha[:usable]
                for j in range(min(len(alpha), attn_edge_index.shape[1])):
                    dst = int(attn_edge_index[1, j])
                    if 0 <= dst < n_nodes:
                        node_attention[dst] += float(alpha[j])
        if node_attention.size and node_attention.sum() > 0:
            node_attention /= node_attention.sum()
        elif node_attention.size:
            node_attention[:] = 1.0 / node_attention.size

        features = seq_meta.get("feature_names", seq_meta.get("feature_list", [])) or []
        id_maps = seq_meta.get("node_id_maps", []) or []
        node_ids = []
        if isinstance(id_maps, dict):
            map_value = id_maps.get("0", id_maps.get(0, id_maps))
            ids = _ordered_ids(map_value, n_nodes)
            node_ids = [ids] if ids else []
        elif isinstance(id_maps, (list, tuple)) and id_maps:
            ids = _ordered_ids(id_maps[0], n_nodes)
            node_ids = [ids] if ids else []
        gmeta = {
            "primary_snapshot": 0,
            "snapshot_ids": seq_meta.get("snapshot_ids", seq_meta.get("window_ids", [])) or [],
            "node_ids": node_ids,
            "feature_names": list(features),
            "attention_aggregation": "first snapshot; node weights are incident GATv2 layer-1 attention aggregated by destination",
            "edge_record_ids": [],  # source export has no edge-to-flow ID map; never invent one
        }
        return {"embedding": h, "node_attention": node_attention, "edge_attention": edge_attention,
                "graph_metadata": gmeta, "raw_member2_attention": raw}


def build_adapter(checkpoint=None, model_config=None, config=None):
    return FoundationModelAdapter.from_checkpoint(checkpoint, model_config)
