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
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.1"
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


def _canonical_label(value: Any) -> str:
    """Normalise the Member-2 package's numeric labels without guessing unknown values."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    if text in {"benign", "normal", "0", "0.0"}:
        return LABEL_BENIGN
    if text in {"attack", "malicious", "1", "1.0"}:
        return LABEL_ATTACK
    return text


def attack_family_tokens(value: Any) -> List[str]:
    """Return individual family names from the package's pipe-separated family field."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    vals = [v.strip() for v in str(value).split("|") if v.strip()]
    return [v for v in vals if v.lower() not in {"benign", "normal", "none", "nan"}]


def has_attack_family(value: Any, family: str) -> bool:
    return str(family).strip() in attack_family_tokens(value)


def _canonicalize_member2_records(df: pd.DataFrame) -> pd.DataFrame:
    """Translate the actual Member-2 export vocabulary to Member-3's internal contract.

    This is format normalisation only: sample IDs, embeddings, source indices and labels are
    not fabricated. Numeric labels are mapped only for the documented 0=benign / 1=attack
    encoding. Unknown labels remain unknown and are rejected by ``validate_records``.
    """
    df = df.copy()
    if "split" in df.columns:
        aliases = {
            "memory_train": SPLIT_MEMORY,
            "train_benign": SPLIT_MEMORY,
            "calibration": SPLIT_CALIBRATION,
            "calibration_benign": SPLIT_CALIBRATION,
            "online_benign": SPLIT_ONLINE_BENIGN,
            "test": SPLIT_TEST,
        }
        df["split"] = df["split"].map(
            lambda v: aliases.get(str(v).strip().lower(), str(v).strip())
        )
    if "label" in df.columns:
        original_labels = df["label"].copy()
        normalised_labels = original_labels.map(_canonical_label)
        if "label_native" not in df.columns and any(
            str(old).strip().lower() != str(new).strip().lower()
            for old, new in zip(original_labels, normalised_labels)
        ):
            df["label_native"] = original_labels.astype(str)
        df["label"] = normalised_labels
        # For normal records, the family is the benign class, not a family named "BENIGN".
        if "attack_family" in df.columns:
            is_benign = df["label"] == LABEL_BENIGN
            df.loc[is_benign, "attack_family"] = LABEL_BENIGN
    if "timestamp" in df.columns:
        # Blank timestamps from the source file stay missing; downstream code must not
        # mistake an empty string for a real timestamp.
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for col in ("dataset_id", "split", "sample_id", "window_id", "attack_family"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def load_records(path: str) -> pd.DataFrame:
    """Read and normalise Member-1/2 embedding records (.npz preferred, .csv accepted)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"record file not found: {path}")
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            if "embedding" not in z.files:
                raise ContractError(f"{path}: npz records must contain an 'embedding' array [N, D]")
            data: Dict[str, Any] = {k: z[k].tolist() for k in z.files if k != "embedding"}
            data["embedding"] = list(z["embedding"].astype(np.float32))
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(path)
        if "embedding" in df.columns:
            emb = np.vstack([np.fromstring(str(v).replace("[", "").replace("]", ""), dtype=np.float32, sep=",")
                             for v in df["embedding"]])
        else:
            # Member-2's CSV export uses h_0 ... h_127 columns rather than one embedding column.
            hcols = sorted((c for c in df.columns if re.fullmatch(r"h_\d+", str(c))),
                           key=lambda c: int(str(c).split("_")[1]))
            if not hcols:
                raise ContractError(f"{path}: expected 'embedding' or h_0...h_D embedding columns")
            emb = df[hcols].to_numpy(dtype=np.float32)
        df = df.drop(columns=[c for c in df.columns if c == "embedding" or re.fullmatch(r"h_\d+", str(c))])
        df["embedding"] = list(emb.astype(np.float32))
    return _canonicalize_member2_records(df)


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

    # Member-2's export explicitly says sample IDs are native sequence_id values, which may
    # repeat between source files. For that export compare dataset_id + sample_id. For the
    # existing handoff contract, IDs are otherwise required to be globally unique.
    dataset_scoped_ids = (
        "sample_id_source" in df.columns
        and df["sample_id_source"].fillna("").astype(str).str.startswith("native:sequence_id").any()
    )

    def sample_identities(sub: pd.DataFrame) -> set:
        if dataset_scoped_ids and "dataset_id" in sub.columns:
            return set(zip(sub["dataset_id"].astype(str), sub["sample_id"].astype(str)))
        return set(sub["sample_id"].astype(str))

    mem = sample_identities(by.get(SPLIT_MEMORY, pd.DataFrame(columns=["sample_id"])))
    cal = sample_identities(by.get(SPLIT_CALIBRATION, pd.DataFrame(columns=["sample_id"])))
    tst = sample_identities(by.get(SPLIT_TEST, pd.DataFrame(columns=["sample_id"])))
    for name, sub in by.items():
        if name in (SPLIT_MEMORY, SPLIT_CALIBRATION) and set(sub["label"]) - {LABEL_BENIGN}:
            raise ContractError(f"CONTAMINATION: split '{name}' contains attack samples")
    if mem & tst:
        raise ContractError(f"CONTAMINATION: {len(mem & tst)} test sample_ids (dataset_id/sample_id) present in memory split")
    if cal & tst:
        raise ContractError(f"CONTAMINATION: {len(cal & tst)} test sample_ids (dataset_id/sample_id) present in calibration split")
    if mem & cal:
        raise ContractError(f"{len(mem & cal)} sample_ids (dataset_id/sample_id) shared by memory and calibration")
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



def assign_stream_metadata(df: pd.DataFrame, window_seconds: float = 30.0) -> pd.DataFrame:
    """Add operational stream fields while preserving source order and native timestamps.

    Unlike ``assign_streams`` (retained for synthetic unit tests), this function never moves
    attacks into fabricated bursts and never fills missing timestamps with invented clock times.
    It sorts by native timestamp when available, otherwise by source_index, otherwise by file row.
    Missing timestamps remain NaN and are recorded as such in the run provenance.
    """
    out = df.copy()
    if "timestamp" not in out.columns:
        out["timestamp"] = np.nan
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    if "dataset_id" not in out.columns:
        out["dataset_id"] = "unknown_dataset"
    if "split" not in out.columns:
        out["split"] = "unknown_split"
    if "label" not in out.columns:
        out["label"] = LABEL_BENIGN
    if "attack_family" not in out.columns:
        out["attack_family"] = LABEL_BENIGN

    out["stream_id"] = ""
    out["stream_pos"] = -1
    if "episode_id" not in out.columns:
        out["episode_id"] = ""
    else:
        out["episode_id"] = out["episode_id"].fillna("").astype(str)

    source_index = pd.to_numeric(out["source_index"], errors="coerce") if "source_index" in out.columns else None
    for (dataset_id, split), group in out.groupby(["dataset_id", "split"], sort=False, dropna=False):
        work = pd.DataFrame({"idx": group.index, "timestamp": group["timestamp"].to_numpy()})
        if source_index is not None:
            work["source_index"] = source_index.loc[group.index].to_numpy()
        # Use real timestamps where the group has complete timestamps; otherwise preserve
        # source-file order. Never synthesise a numeric timestamp from stream position.
        if len(group) and group["timestamp"].notna().all():
            order = work.sort_values(["timestamp"] + (["source_index"] if source_index is not None else []),
                                     kind="mergesort")["idx"].tolist()
            ordering_source = "native_timestamp"
        elif source_index is not None and work["source_index"].notna().all():
            order = work.sort_values("source_index", kind="mergesort")["idx"].tolist()
            ordering_source = "source_index"
        else:
            order = list(group.index)
            ordering_source = "source_row_order"

        stream = f"{dataset_id}:{split}"
        for pos, idx in enumerate(order):
            out.loc[idx, "stream_id"] = stream
            out.loc[idx, "stream_pos"] = pos

        # Only derive episode boundaries from the observed ordered labels. A separate episode
        # starts after benign traffic or when the family set changes.
        episode_no = 0
        previous_key = None
        for idx in order:
            label = _canonical_label(out.at[idx, "label"])
            if label != LABEL_ATTACK:
                out.loc[idx, "episode_id"] = "benign"
                previous_key = None
                continue
            fams = sorted(attack_family_tokens(out.at[idx, "attack_family"]))
            family_key = "|".join(fams) if fams else "unknown_attack_family"
            if previous_key != family_key:
                episode_no += 1
            out.loc[idx, "episode_id"] = f"{family_key}#{episode_no}"
            previous_key = family_key
        out.attrs.setdefault("stream_order_sources", {})[stream] = ordering_source
    out["stream_pos"] = pd.to_numeric(out["stream_pos"], errors="raise").astype(np.int64)
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


def _torch_graph_file_to_dict(path: str) -> Dict[str, Dict[str, Any]]:
    """Load Member-1's native torch/PyG export into the portable Member-3 graph schema."""
    try:
        import torch
    except ImportError as exc:
        raise ContractError("loading a .pt graph file requires the Member-2 PyTorch environment") from exc
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sequences = blob.get("sequences") if isinstance(blob, dict) else None
    if sequences is None:
        raise ContractError(f"{path}: expected a dict with a 'sequences' list")
    top_features = blob.get("feature_list", []) if isinstance(blob, dict) else []
    basename = os.path.basename(path).lower()
    if "graph_validation_v2" in basename:
        source_dataset_id = "dataset2_graph_validation_v2"
    elif basename == "graph.pt" or basename.startswith("graph_train"):
        source_dataset_id = "dataset1_graph_pt"
    else:
        source_dataset_id = str(blob.get("dataset", "")) if isinstance(blob, dict) else ""
    out: Dict[str, Dict[str, Any]] = {}
    for index, seq in enumerate(sequences):
        sid = seq.get("sequence_id", seq.get("sample_id", index))
        sid = str(sid)
        snapshots = []
        for snap in seq.get("snapshots", []):
            def as_numpy(v, dtype=None):
                if v is None:
                    return np.asarray([], dtype=dtype)
                if hasattr(v, "detach"):
                    v = v.detach().cpu().numpy()
                arr = np.asarray(v)
                return arr.astype(dtype, copy=False) if dtype is not None else arr
            x = as_numpy(getattr(snap, "x", None), np.float32)
            edge_index = as_numpy(getattr(snap, "edge_index", None), np.int64).reshape(2, -1)
            edge_attr = as_numpy(getattr(snap, "edge_attr", None), np.float32)
            sd = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}
            for attr in ("y", "edge_y"):
                if hasattr(snap, attr):
                    sd[attr] = as_numpy(getattr(snap, attr))
            snapshots.append(sd)
        graph_record = {
            "sample_id": sid,
            "dataset_id": str(seq.get("dataset_id") or source_dataset_id),
            "window_id": str(seq.get("window_id", seq.get("window_ids", ""))),
            "timestamp": seq.get("timestamp", seq.get("start_time")),
            "snapshot_ids": seq.get("snapshot_ids", seq.get("window_ids", [])),
            "snapshots": snapshots,
            "node_id_maps": _jsonable(seq.get("node_id_maps", [])),
            "feature_names": _jsonable(seq.get("feature_list", top_features)),
        }
        out[sid] = graph_record
        dataset_key = str(graph_record.get("dataset_id") or source_dataset_id)
        if dataset_key:
            out[f"{dataset_key}::{sid}"] = graph_record
    return out


def load_graphs(path: str) -> Dict[str, Dict[str, Any]]:
    """Load graph sequences keyed by sample_id from JSON/NPZ/CSV or native PyTorch .pt.

    Native .pt support is optional and requires torch in the active environment. Records still
    score from cached embeddings; graph loading is only needed for graph explainability/Fidelity+.
    """
    if isinstance(path, (list, tuple)):
        merged: Dict[str, Dict[str, Any]] = {}
        for item in path:
            if item and os.path.exists(str(item)):
                # Dataset-qualified keys stay unique. The unqualified key is intentionally
                # last-file-wins for older callers; dataset-aware callers should use dset::id.
                merged.update(load_graphs(str(item)))
        return merged
    path = str(path)
    if path.endswith(".pt") or path.endswith(".pth"):
        return _torch_graph_file_to_dict(path)
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return {str(k): (json.loads(v) if isinstance(v, str) else v) for k, v in raw.items()}
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if "graph_json" not in df.columns or "sample_id" not in df.columns:
            raise ContractError("graph csv needs 'sample_id' and 'graph_json' columns (contract §E)")
        return {str(s_): json.loads(g) for s_, g in zip(df["sample_id"], df["graph_json"])}
    if not path.endswith(".npz"):
        raise ContractError(f"unsupported graph file '{path}': expected .pt/.pth/.json/.npz or .csv")
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
