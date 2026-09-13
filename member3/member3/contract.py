"""Member-2 handoff contract: adapter loading, validation, record I/O, masking.

This module is the *only* place where Member 3 touches upstream artefacts.
It implements no foundation-model logic: the real encoder lives in Member 2
and is reached through ``FoundationAdapter`` (see docs/MEMBER2_HANDOFF_CONTRACT.md).

Contents
--------
* ``load_config`` / ``set_seed`` / ``l2_normalize``      - tiny shared helpers
* ``load_adapter``                                       - import + validate Member 2 adapter
* ``load_records`` / ``save_records``                    - embedding-record I/O
* ``validate_records`` / ``assert_splits_clean``         - leakage + schema guards
* ``mask_graph_sequence``                                - node/edge masking for Fidelity+
Nothing here generates data, and there is no stand-in encoder. Member 3 has exactly one
adapter source: the file Member 2 hands over (docs/MEMBER2_HANDOFF_CONTRACT.md). If it is
missing the run aborts.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"
EPS_DEFAULT = 1e-12
RECORD_COLUMNS = [
    "dataset_id",
    "split",
    "sample_id",
    "window_id",
    "timestamp",
    "embedding",
    "label",
    "attack_family",
]
SPLIT_MEMORY = "train_benign"
SPLIT_CALIBRATION = "calibration_benign"
SPLIT_TEST = "test"
SPLIT_ONLINE_BENIGN = "online_benign"
LABEL_BENIGN = "benign"
LABEL_ATTACK = "attack"


class ContractError(RuntimeError):
    """Raised when a Member-2 artefact violates the handoff contract."""


# --------------------------------------------------------------------------- #
# configuration / determinism
# --------------------------------------------------------------------------- #
def load_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "member2" not in cfg:
        raise ContractError(f"{path}: missing top-level key 'member2'")
    return cfg


def set_seed(seed: int) -> None:
    """Deterministic RNG for every randomised Member-3 component."""
    np.random.seed(int(seed))
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    try:  # only relevant when the Member-2 adapter needs torch
        import torch

        torch.manual_seed(int(seed))
    except Exception:
        pass


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x / max(float(np.linalg.norm(x)), eps)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def software_versions() -> Dict[str, str]:
    out: Dict[str, str] = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}
    for mod in ("faiss", "sklearn", "matplotlib", "scipy", "torch", "fastapi", "yaml"):
        try:
            m = importlib.import_module(mod)
            out[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception:
            out[mod] = "not-installed"
    return out


# --------------------------------------------------------------------------- #
# Member-2 adapter
# --------------------------------------------------------------------------- #
class FoundationAdapter:
    """Required Member-2 interface (documentation of the duck-type contract).

    Methods
    -------
    encode(graph_sequence)            -> np.ndarray, shape (D,) or (B, D)
    encode_with_attention(graph_sequence) -> dict with keys
        ``embedding``, ``node_attention``, ``edge_attention``, ``graph_metadata``
    metadata()                        -> dict (see docs/MEMBER2_HANDOFF_CONTRACT.md §A)

    Optional (needed only for node/edge-masking Fidelity+):
    mask_nodes(graph_sequence, node_ids) / mask_edges(graph_sequence, edge_ids)
    """

    required = ("encode", "encode_with_attention", "metadata")


def _load_py_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ContractError(f"cannot import python module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_adapter(cfg: Dict[str, Any], artifacts_dir: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    """Return ``(adapter, adapter_meta)`` honouring the handoff contract.

    ``member2.adapter_path`` if set, else ``<member2.artifacts_dir>/model_adapter.py``.
    There is no fallback of any kind: a missing or non-conforming adapter aborts the run,
    because silently substituting anything else would invalidate every number downstream.
    """
    m2 = dict(cfg.get("member2", {}))
    artifacts_dir = artifacts_dir or m2.get("artifacts_dir", "member2_artifacts")
    adapter_path = m2.get("adapter_path") or os.path.join(artifacts_dir, "model_adapter.py")
    embedding_dim = int(m2.get("embedding_dim", 0) or 0)

    if not os.path.exists(adapter_path):
        raise ContractError(
            f"Member-2 adapter not found at '{adapter_path}'.\n"
            "Deliver the artefacts specified in docs/MEMBER2_HANDOFF_CONTRACT.md "
            "(model_adapter.py + checkpoint.pt + model_config.yaml) and set "
            "member2.artifacts_dir (or member2.adapter_path) accordingly.")

    mod = _load_py_module(adapter_path, "member3_member2_adapter")
    factory = getattr(mod, "FoundationModelAdapter", None) or getattr(mod, "build_adapter", None)
    if factory is None:
        raise ContractError(
            f"{adapter_path}: expected class 'FoundationModelAdapter' or function 'build_adapter'")
    ckpt = os.path.join(artifacts_dir, m2.get("checkpoint", "checkpoint.pt"))
    mcfg = os.path.join(artifacts_dir, m2.get("model_config", "model_config.yaml"))
    adapter = (factory.from_checkpoint(ckpt, mcfg) if hasattr(factory, "from_checkpoint")
               else factory(checkpoint=ckpt, model_config=mcfg, config=m2))
    origin = "member2_adapter"
    meta = validate_adapter(adapter, expected_dim=embedding_dim)
    meta["adapter_origin"] = origin
    meta["adapter_path"] = adapter_path if os.path.exists(adapter_path) else None
    return adapter, meta


def validate_adapter(adapter: Any, expected_dim: int = 0) -> Dict[str, Any]:
    for name in FoundationAdapter.required:
        if not hasattr(adapter, name) or not callable(getattr(adapter, name)):
            raise ContractError(f"Member-2 adapter missing required method '{name}()' (see handoff contract §B/§C)")
    meta = dict(getattr(adapter, "metadata")() or {})
    dim = int(meta.get("embedding_dim", 0) or 0)
    if dim <= 0:
        raise ContractError("Member-2 adapter metadata() must report a positive 'embedding_dim'")
    if expected_dim and dim != expected_dim:
        raise ContractError(f"embedding dimension mismatch: member2 says {dim}, config.yaml says {expected_dim}")
    if not meta.get("l2_normalized", False):
        # tolerated: Member 3 normalises and records who did it (see §9 scoring definition)
        meta["l2_normalized"] = False
    return meta


# --------------------------------------------------------------------------- #
# embedding records
# --------------------------------------------------------------------------- #
def _np_safe(arr: np.ndarray) -> np.ndarray:
    """Store object columns as native npz arrays (no pickle => portable, loadable)."""
    if arr.dtype != object:
        return arr
    vals = [v.item() if hasattr(v, "item") else v for v in arr]
    if all(isinstance(v, (int, float, np.integer, np.floating)) or v is None for v in vals):
        return np.asarray([np.nan if v is None else v for v in vals], dtype=np.float64)
    return np.asarray(["" if v is None else str(v) for v in vals], dtype="U")


def save_records(df: pd.DataFrame, path: str) -> None:
    """Store Member-2 embeddings + evaluation metadata reproducibly (.npz or .csv)."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    emb = np.vstack([np.asarray(e, dtype=np.float32).ravel() for e in df["embedding"]])
    if path.endswith(".npz"):
        arrays: Dict[str, Any] = {"embedding": emb}
        for c in df.columns:
            if c == "embedding":
                continue
            arrays[c] = _np_safe(df[c].to_numpy())
        np.savez_compressed(path, **arrays)
    else:
        out = df.copy()
        out["embedding"] = [",".join(f"{v:.6f}" for v in e) for e in emb]
        out.to_csv(path, index=False)


def load_records(path: str) -> pd.DataFrame:
    """Read the embedding records produced by Member 1/2 (see handoff contract)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"record file not found: {path}")
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            if "embedding" not in z.files:
                raise ContractError(f"{path}: npz records must contain an 'embedding' array [N, D]")
            data: Dict[str, Any] = {k: z[k].tolist() for k in z.files if k != "embedding"}
            data["embedding"] = list(z["embedding"].astype(np.float32))
        return pd.DataFrame(data)
    df = pd.read_csv(path)
    if "embedding" not in df.columns:
        raise ContractError(f"{path}: CSV records must contain an 'embedding' column")
    emb = np.vstack([np.fromstring(str(s).replace("[", "").replace("]", ""), dtype=np.float32, sep=",")
                     for s in df["embedding"]])
    df["embedding"] = list(emb)
    return df


def validate_records(df: pd.DataFrame, expected_dim: Optional[int] = None, need_labels: bool = True) -> None:
    missing = [c for c in RECORD_COLUMNS if c not in df.columns]
    if "attack_family" in missing and not need_labels:
        missing.remove("attack_family")
    if "label" in missing and not need_labels:
        missing.remove("label")
    if missing:
        raise ContractError(f"record file is missing required columns: {missing}")
    if len(df) == 0:
        raise ContractError("record file is empty")
    dims = {len(e) for e in df["embedding"]}
    if len(dims) != 1:
        raise ContractError(f"ragged embedding dimensionality in records: {sorted(dims)}")
    d = dims.pop()
    if expected_dim and d != int(expected_dim):
        raise ContractError(f"embedding dimension mismatch: records carry {d}, Member-2 declares {expected_dim}")
    if need_labels:
        bad = set(df["label"].astype(str)) - {LABEL_BENIGN, LABEL_ATTACK}
        if bad:
            raise ContractError(f"label column must be 'benign'/'attack'; found {sorted(bad)}")


def assert_splits_clean(df: pd.DataFrame, permitted_reference_datasets: Optional[Sequence[str]] = None) -> None:
    """Contamination guard (§29).  Runs before every stage, on real data.

    Invariants enforced:
      * memory split  : benign only, sample_id disjoint from test split
      * calibration   : benign only, sample_id disjoint from test split
      * test split    : may contain attacks; must not overlap memory/calibration ids
    """
    by = {s: g for s, g in df.groupby("split")}
    mem = set(by.get(SPLIT_MEMORY, pd.DataFrame(columns=["sample_id"]))["sample_id"])
    cal = set(by.get(SPLIT_CALIBRATION, pd.DataFrame(columns=["sample_id"]))["sample_id"])
    tst = set(by.get(SPLIT_TEST, pd.DataFrame(columns=["sample_id"]))["sample_id"])
    for name, sub in by.items():
        if name in (SPLIT_MEMORY, SPLIT_CALIBRATION) and set(sub["label"]) - {LABEL_BENIGN}:
            raise ContractError(f"CONTAMINATION: split '{name}' contains attack samples")
    if mem & tst:
        raise ContractError(f"CONTAMINATION: {len(mem & tst)} test sample_ids present in memory split")
    if cal & tst:
        raise ContractError(f"CONTAMINATION: {len(cal & tst)} test sample_ids present in calibration split")
    if mem & cal:
        raise ContractError(f"{len(mem & cal)} sample_ids shared by memory and calibration")
    if permitted_reference_datasets:
        allowed = {str(d) for d in permitted_reference_datasets}
        for name in (SPLIT_MEMORY, SPLIT_CALIBRATION):
            sub = by.get(name, pd.DataFrame(columns=["dataset_id"]))
            bad_ds = sorted({str(x) for x in sub["dataset_id"].unique()} - allowed) if len(sub) else []
            if bad_ds:
                raise ContractError(
                    f"CONTAMINATION: reference split '{name}' draws from dataset(s) {bad_ds} which are not "
                    f"permitted calibration sources {sorted(allowed)} (test dataset must never seed the memory)")


# --------------------------------------------------------------------------- #
# graph-sequence masking (Fidelity+ recompute)
# --------------------------------------------------------------------------- #
def mask_graph_sequence(seq: Dict[str, Any], node_ids: Optional[Sequence[int]] = None,
                        edge_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Causally mask a temporal graph sequence: drop nodes (and incident edges)
    or zero the flow features of selected edges.

    Accepted sequence format (handoff contract §E) - a dict with
    ``snapshots``: list of dicts holding ``x`` [n,F], ``edge_index`` [2,E],
    ``edge_attr`` [E,F].
    """
    if not isinstance(seq, dict) or "snapshots" not in seq:
        raise ContractError(
            "mask_graph_sequence expects a dict with key 'snapshots' (contract §E). "
            "Provide adapter.mask_nodes/mask_edges if Member 1 uses another graph object."
        )
    node_ids = set(int(i) for i in (node_ids or []))
    edge_ids = set(int(i) for i in (edge_ids or []))
    if not node_ids and not edge_ids:
        raise ContractError("mask_graph_sequence called with nothing to mask")
    out = dict(seq)
    snaps = []
    for snap in seq["snapshots"]:
        s = {k: v for k, v in snap.items()}
        ei = np.asarray(snap["edge_index"], dtype=np.int64).reshape(2, -1)
        if node_ids:
            n = int(np.asarray(snap["x"]).shape[0])
            keep_n = np.array([i for i in range(n) if i not in node_ids], dtype=np.int64)
            remap = -np.ones(n, dtype=np.int64)
            remap[keep_n] = np.arange(len(keep_n))
            eattr = np.asarray(snap.get("edge_attr"))
            if ei.shape[1]:
                keep_e = np.all(np.isin(ei.T, keep_n), axis=1)
                ei_new = remap[ei[:, keep_e]] if keep_e.any() else np.zeros((2, 0), dtype=np.int64)
                if eattr.size:
                    eattr = eattr[keep_e]
            else:
                ei_new, eattr = ei, eattr
            s["x"] = np.asarray(snap["x"], dtype=np.float32)[keep_n] if len(keep_n) else np.zeros((0, np.asarray(snap['x']).shape[1]), np.float32)
            s["edge_index"] = ei_new
            if np.asarray(snap.get("edge_attr")).size:
                s["edge_attr"] = eattr
            s["masked_nodes"] = sorted(int(i) for i in node_ids)
        if edge_ids:
            eattr = np.asarray(snap["edge_attr"], dtype=np.float32).copy()
            for j in edge_ids:
                if 0 <= j < eattr.shape[0]:
                    eattr[j] = 0.0
            s["edge_attr"] = eattr
            s["masked_edges"] = sorted(int(i) for i in edge_ids)
        snaps.append(s)
    out["snapshots"] = snaps
    return out


def assign_streams(df: pd.DataFrame, window_seconds: float = 30.0, seed: int = 0) -> pd.DataFrame:
    """Lay the test samples out as an ordered live stream with attack *bursts*.

    Detection delay and the multi-stage confirmation rule are only meaningful for a
    temporally ordered stream, so benign windows form the background and each attack
    family enters as contiguous bursts (episode_id marks the burst).
    """
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    df["stream_id"] = ""
    df["stream_pos"] = -1
    df["episode_id"] = ""
    for dataset_id, g in df.groupby("dataset_id"):
        g = g.copy()
        order: List[int] = []
        episodes: Dict[int, str] = {}
        ben = list(g.index[g["label"] == LABEL_BENIGN])
        cursor = 0
        order.extend(int(i) for i in ben)
        attacks = g[g["label"] == LABEL_ATTACK]
        bursts: List[Tuple[int, List[int]]] = []
        for fam in sorted(attacks["attack_family"].unique()):
            rows = list(attacks.index[attacks["attack_family"] == fam])
            size = max(3, min(len(rows), int(rng.integers(4, 12))))
            for s0, j in enumerate(range(0, len(rows), size)):
                chunk = rows[j : j + size]
                if len(chunk) >= 2:
                    pos = int(rng.integers(0, len(order) + 1))
                    bursts.append((pos, chunk))
                    for c in chunk:
                        episodes[c] = f"{fam}#{s0}"
        for pos, chunk in sorted(bursts, key=lambda b: b[0], reverse=True):
            order = order[:pos] + list(chunk) + order[pos:]
        for i, idx in enumerate(order):
            df.loc[idx, "stream_pos"] = i
            df.loc[idx, "timestamp"] = float(i) * float(window_seconds)
            df.loc[idx, "stream_id"] = f"{dataset_id}:test" if df.loc[idx, "split"] == SPLIT_TEST else f"{dataset_id}:{df.loc[idx, 'split']}"
            df.loc[idx, "episode_id"] = episodes.get(idx, "benign")
        leftover = [i for i in g.index if df.loc[i, "stream_pos"] == -1]
        for k, idx in enumerate(leftover):
            df.loc[idx, "stream_pos"] = len(order) + k
            df.loc[idx, "timestamp"] = float(len(order) + k) * float(window_seconds)
            df.loc[idx, "stream_id"] = f"{dataset_id}:{df.loc[idx, 'split']}"
            df.loc[idx, "episode_id"] = "benign"
    return df


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_graphs(path: str) -> Dict[str, Dict[str, Any]]:
    """Temporal-graph sequences keyed by ``sample_id`` (contract §E).

    Accepted shapes, in this order: ``.json`` mapping of ``sample_id`` -> sequence; ``.npz``
    with one JSON string per key; ``.csv`` with a ``graph_json`` column.  Used only for
    masking-and-recompute (Fidelity+); scoring itself reads cached embeddings.
    """
    path = str(path)
    if path.endswith(".json"):
        with open(path) as fh:
            raw = json.load(fh)
        return {str(k): (json.loads(v) if isinstance(v, str) else v) for k, v in raw.items()}
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if "graph_json" not in df.columns or "sample_id" not in df.columns:
            raise ContractError("graph csv needs 'sample_id' and 'graph_json' columns (contract §E)")
        return {str(s_): json.loads(g) for s_, g in zip(df["sample_id"], df["graph_json"])}
    if not path.endswith(".npz"):
        raise ContractError(f"unsupported graph file '{path}': expected .json, .npz or .csv")
    out: Dict[str, Dict[str, Any]] = {}
    with np.load(path, allow_pickle=True) as z:
        for k in z.files:
            v = z[k]
            v = v[0] if getattr(v, "shape", None) and v.ndim and v.shape[0] == 1 else v
            out[str(k)] = json.loads(str(v)) if isinstance(v, (str, np.str_)) else (
                v.item() if hasattr(v, "item") else v)
    return out


def save_graphs(graphs: Dict[str, Dict[str, Any]], path: str) -> None:
    """Write a graph-sequence file in the accepted format (used by the test-suite; upstream
    members may write any of the three shapes above)."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    payload = {str(k): json.dumps(_jsonable(v)) for k, v in graphs.items()}
    if path.endswith(".json"):
        with open(path, "w") as fh:
            json.dump(payload, fh)
    else:
        np.savez_compressed(path, **{k: np.array([v]) for k, v in payload.items()})


def git_commit(cwd: str = ".") -> str:
    """Git revision of the run, or 'unavailable' - never guessed."""
    try:
        import subprocess

        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "not-a-git-repo"
    except Exception:
        return "unavailable"
