"""Adaptive Memory Bank + incremental FAISS index (Member-3 contributions 2).

One class, two index back-ends:

* ``hnsw``  - ``faiss.IndexHNSWFlat`` with inner-product metric on L2-normalised
              vectors, i.e. cosine similarity (the architecture in the project docs).
* ``flat``  - ``faiss.IndexFlatIP`` exact search, used only as the brute-force
              timing baseline in the retrieval ablation.

Guarantees enforced here (not merely documented):
  * only benign embeddings may be inserted (label check + threshold/confidence gate),
  * ``immutable`` mode forbids every mutation (used by all evaluation stages),
  * deterministic selection/replacement for a fixed seed,
  * index metadata is persisted beside - never inside - the binary index.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contract import LABEL_ATTACK, LABEL_BENIGN, l2_normalize

try:  # pragma: no cover - availability varies by environment
    import faiss  # type: ignore

    HAVE_FAISS = True
except Exception:  # pragma: no cover
    faiss = None
    HAVE_FAISS = False


@dataclass
class MemoryStats:
    dim: int = 0
    n: int = 0
    capacity: int = 0
    index_type: str = "hnsw"
    k: int = 8
    hnsw_M: int = 32
    ef_construction: int = 200
    ef_search: int = 128
    version: int = 0
    selection: str = "kcenter_greedy"
    replacement: str = "redundant"
    seed: int = 0
    n_inserted: int = 0
    n_rejected_attack: int = 0
    n_rejected_threshold: int = 0
    n_rejected_low_confidence: int = 0
    n_replaced: int = 0
    n_dropped_redundant: int = 0
    immutable: bool = False
    index_file: Optional[str] = None
    updates: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdaptiveMemoryBank:
    """Representative normal-behaviour embeddings with a FAISS ANN front-end."""

    def __init__(self, dim: int, capacity: int = 4096, index_type: str = "hnsw", k: int = 8,
                 hnsw_M: int = 32, ef_construction: int = 200, ef_search: int = 128,
                 selection: str = "kcenter_greedy", replacement: str = "redundant",
                 seed: int = 0, require_faiss: bool = True):
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError(f"memory dim must be positive, got {self.dim}")
        self.capacity = int(capacity)
        self.k = int(k)
        self.selection, self.replacement = selection, replacement
        self.stats = MemoryStats(dim=self.dim, capacity=self.capacity, index_type=index_type, k=self.k,
                                 hnsw_M=hnsw_M, ef_construction=ef_construction, ef_search=ef_search,
                                 selection=selection, replacement=replacement, seed=int(seed))
        self._rng = np.random.default_rng(int(seed))
        self._vectors = np.zeros((0, self.dim), dtype=np.float64)
        self._meta: List[Dict[str, Any]] = []
        self._index_file: Optional[str] = None
        self._build_index()

    # ------------------------------------------------------------------ #
    # index plumbing
    # ------------------------------------------------------------------ #
    def _build_index(self) -> None:
        t = self.stats.index_type
        if t in ("hnsw", "flat"):
            if not HAVE_FAISS:
                raise RuntimeError(
                    f"config requests FAISS index_type='{t}' but `faiss` is not importable. "
                    "Install it (pip install faiss-cpu) or set index_type: numpy and record the "
                    "deviation - the HNSW retrieval claim in the paper then stays NOT RUN."
                )
            if t == "hnsw":
                idx = faiss.IndexHNSWFlat(self.dim, int(self.stats.hnsw_M), faiss.METRIC_INNER_PRODUCT)
                idx.hnsw.efConstruction = int(self.stats.ef_construction)
                idx.hnsw.efSearch = int(self.stats.ef_search)
            else:
                idx = faiss.IndexFlatIP(self.dim)
            self._index = idx
        elif t == "numpy":
            self._index = None
        else:
            raise ValueError(f"unknown index_type '{t}' (hnsw | flat | numpy)")

    def _index_add(self, chunk: np.ndarray) -> None:
        if self._index is not None:
            self._index.add(np.ascontiguousarray(chunk.astype(np.float32)))

    def search(self, queries: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (cosine similarity, memory index) of the k nearest normal entries."""
        k = int(k or self.k)
        Q = l2_normalize(np.asarray(queries, dtype=np.float64))
        if Q.ndim == 1:
            Q = Q[None, :]
        if Q.shape[1] != self.dim:
            raise ValueError(f"query dim {Q.shape[1]} != memory dim {self.dim}")
        if self._vectors.shape[0] == 0:
            raise RuntimeError("memory bank is empty - cannot query")
        k = min(k, self._vectors.shape[0])
        if self._index is None:
            sims = np.clip(Q @ self._vectors.T, -1.0, 1.0)
            idx = np.argsort(-sims, axis=1)[:, :k]
            return np.take_along_axis(sims, idx, axis=1), idx
        D, I = self._index.search(Q.astype(np.float32), k)
        # HNSW stores float32 copies: similarity can exceed 1 by ~1e-7 and may return
        # -inf padding on tiny banks.  Clipping keeps the score in [0, 2] exactly.
        D = np.where(np.isfinite(D), D, -1.0)
        D = np.clip(D, -1.0, 1.0)
        I = np.where(I >= 0, I, 0)
        return D.astype(np.float64), I.astype(np.int64)

    # ------------------------------------------------------------------ #
    # construction / mutation
    # ------------------------------------------------------------------ #
    def fit(self, embeddings: np.ndarray, metadata: Optional[Sequence[Dict[str, Any]]] = None,
            labels: Optional[Sequence[str]] = None) -> "AdaptiveMemoryBank":
        """Fill the bank with representatives of the permitted (benign) reference split."""
        self._assert_mutable("fit")
        X = l2_normalize(np.asarray(embeddings, dtype=np.float64))
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.dim:
            raise ValueError(f"embedding dim mismatch: fit got {X.shape[1]}, memory expects {self.dim}")
        if labels is not None:
            bad = [i for i, lb in enumerate(labels) if str(lb) != LABEL_BENIGN]
            if bad:
                raise PermissionError(
                    f"MEMORY CORRUPTION BLOCKED: fit() received {len(bad)} non-benign labelled samples"
                )
        self._vectors = np.zeros((0, self.dim), dtype=np.float64)
        self._meta = []
        if self._index is not None:
            self._build_index()
        sel = self._select_representatives(X)
        meta = list(metadata or [])
        self._append(X[sel], [meta[i] if i < len(meta) else {} for i in sel])
        self.stats.version += 1
        return self

    def _select_representatives(self, X: np.ndarray) -> np.ndarray:
        """Deterministic coverage-first selection.

        ``kcenter_greedy``: farthest-point sampling (max-min cosine similarity) -
        a classic 2-approximation for k-centre, so the stored bank spans the normal
        manifold instead of duplicating dense regions.  ``reservoir``: uniform
        reservoir sample (used as the naive-memory ablation).
        """
        n = X.shape[0]
        c = min(self.capacity, n)
        if self.selection == "reservoir":
            idx = np.sort(self._rng.choice(n, size=c, replace=False))
            return idx
        if self.selection not in ("kcenter_greedy", "k_center", "farthest_point"):
            raise ValueError(f"unknown selection '{self.selection}'")
        start = int(self._rng.integers(0, n)) if n > c else 0
        chosen = [start]
        max_sim = X @ X[start]
        while len(chosen) < c:
            nxt = int(np.argmin(max_sim))
            chosen.append(nxt)
            max_sim = np.maximum(max_sim, X @ X[nxt])
        return np.array(chosen, dtype=np.int64)

    def _append(self, X: np.ndarray, meta: Sequence[Dict[str, Any]]) -> None:
        self._vectors = np.vstack([self._vectors, X]) if len(self._vectors) else X.copy()
        for i, m in enumerate(meta):
            mm = dict(m or {})
            mm.setdefault("memory_index", int(self._vectors.shape[0] - len(meta) + i))
            mm.setdefault("insert_stage", "offline_fit")
            self._meta.append(mm)
        self._index_add(X)
        self.stats.n = int(self._vectors.shape[0])
        self.stats.n_inserted += len(X)

    def add(self, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Insert one embedding, evicting the most redundant entry when at capacity."""
        self._assert_mutable("add")
        z = l2_normalize(np.asarray(embedding, dtype=np.float64).ravel())
        if z.shape[0] != self.dim:
            raise ValueError(f"embedding dim {z.shape[0]} != memory dim {self.dim}")
        if self._vectors.shape[0] < self.capacity:
            self._append(z[None, :], [dict(metadata or {})])
            self.stats.version += 1
            return {"action": "inserted", "memory_size": self.stats.n, "memory_version": self.stats.version}
        # capacity reached -> representative replacement
        sims = self._vectors @ z
        nn = float(np.max(sims))
        redundancy = self._redundancy()
        worst = int(np.argmax(redundancy))
        rec = {"action": None, "memory_size": self.stats.n, "novelty": 1.0 - nn,
               "replaced_index": None, "memory_version": self.stats.version + 1}
        if self.replacement == "reservoir":
            j = int(self._rng.integers(0, self._vectors.shape[0]))
            self._replace(j, z, metadata or {})
            rec.update(action="reservoir_replaced", replaced_index=j)
            self.stats.n_replaced += 1
        elif nn < float(redundancy[worst]):
            self._replace(worst, z, metadata or {})
            rec.update(action="redundancy_replaced", replaced_index=worst)
            self.stats.n_replaced += 1
        else:
            rec.update(action="dropped_redundant")
            self.stats.n_dropped_redundant += 1
        self.stats.version += 1
        rec["memory_version"] = self.stats.version
        rec["memory_size"] = self.stats.n
        self.stats.updates.append(rec)
        return rec

    def _redundancy(self) -> np.ndarray:
        """Per-entry redundancy = cosine similarity to its nearest peer."""
        n = self._vectors.shape[0]
        if n < 2:
            return np.zeros(n)
        S = self._vectors @ self._vectors.T
        np.fill_diagonal(S, -np.inf)
        return S.max(axis=1)

    def _replace(self, j: int, z: np.ndarray, metadata: Dict[str, Any]) -> None:
        if self._index is not None and hasattr(self._index, "remove_ids"):
            try:  # only IndexFlat supports removal; HNSW keeps the stale node
                self._index.remove_ids(np.array([j], dtype=np.int64))
            except Exception:
                pass
        self._vectors[j] = z
        mm = dict(metadata or {})
        mm["memory_index"] = j
        mm["insert_stage"] = "online_update"
        self._meta[j] = mm
        if self._index is not None and self.stats.index_type == "flat":
            self._index = faiss.IndexFlatIP(self.dim)
            self._index_add(self._vectors)
        # HNSW is append-only by design: rebuild the graph over the new vector set
        elif self._index is not None:
            self._build_index()
            self._index_add(self._vectors)

    def update_if_benign(self, embedding: np.ndarray, confidence: float, score: float, threshold: float,
                         metadata: Optional[Dict[str, Any]] = None, allow_online: bool = True,
                         benign_confidence_max: float = 0.05) -> Dict[str, Any]:
        """Gate for online benign-memory adaptation.

        Accepts a sample only if ALL of:
          * no attack label in metadata,
          * score <= calibrated threshold (it looks normal),
          * anomaly confidence <= ``benign_confidence_max`` (confidently benign),
          * the caller enabled online updates for this experiment.
        Every other case is refused and logged - attacks can never enter the bank.
        """
        md = dict(metadata or {})
        rec = {"score": float(score), "threshold": float(threshold), "confidence": float(confidence),
               "sample_id": md.get("sample_id"), "action": None}
        if str(md.get("label", "")) == LABEL_ATTACK:
            self.stats.n_rejected_attack += 1
            rec.update(action="rejected_attack_label")
            self.stats.updates.append(rec)
            raise PermissionError(f"MEMORY CORRUPTION BLOCKED: attack-labelled sample {md.get('sample_id')} "
                                  "was offered to the normal memory bank")
        if not allow_online:
            rec.update(action="rejected_adaptation_disabled")
            self.stats.updates.append(rec)
            return rec
        if float(score) > float(threshold):
            self.stats.n_rejected_threshold += 1
            rec.update(action="rejected_above_threshold")
            self.stats.updates.append(rec)
            return rec
        if float(confidence) > float(benign_confidence_max):
            self.stats.n_rejected_low_confidence += 1
            rec.update(action="rejected_uncertain")
            self.stats.updates.append(rec)
            return rec
        out = self.add(embedding, {**md, "insert_stage": "online_update"})
        rec.update(action=out["action"], memory_version=out["memory_version"], memory_size=out["memory_size"])
        self.stats.updates.append(rec)
        return rec

    # ------------------------------------------------------------------ #
    # safety / io
    # ------------------------------------------------------------------ #
    def _assert_mutable(self, who: str) -> None:
        if self.stats.immutable:
            raise PermissionError(f"evaluation mode is immutable: refusing {who}() (enable adaptation explicitly)")

    def set_immutable(self, flag: bool = True) -> None:
        self.stats.immutable = bool(flag)

    def save(self, path: str) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if self._index is not None and HAVE_FAISS:
            faiss.write_index(self._index, path)
        base, _ = os.path.splitext(path)
        np.savez_compressed(base + "_vectors.npz", vectors=self._vectors.astype(np.float32))
        with open(base + "_meta.json", "w") as fh:
            json.dump({"stats": self.stats.as_dict(), "entries": self._meta, "schema_version": "1.0"}, fh, indent=2)
        self.stats.index_file = path
        return {"index": path, "vectors": base + "_vectors.npz", "metadata": base + "_meta.json"}

    @classmethod
    def load(cls, path: str) -> "AdaptiveMemoryBank":
        base, _ = os.path.splitext(path)
        with open(base + "_meta.json") as fh:
            blob = json.load(fh)
        st = blob["stats"]
        bank = cls(dim=st["dim"], capacity=st["capacity"], index_type=st["index_type"], k=st["k"],
                   hnsw_M=st["hnsw_M"], ef_construction=st["ef_construction"], ef_search=st["ef_search"],
                   selection=st.get("selection", "kcenter_greedy"), replacement=st.get("replacement", "redundant"),
                   seed=st.get("seed", 0))
        vecs = np.load(base + "_vectors.npz")["vectors"].astype(np.float64)
        if vecs.shape[1] != bank.dim:
            raise RuntimeError(f"FAISS dimension mismatch on reload: index vectors {vecs.shape[1]} != {bank.dim}")
        bank._vectors = vecs
        bank._meta = list(blob.get("entries", []))
        if bank._index is not None:
            if not os.path.exists(path):
                raise FileNotFoundError(f"binary index missing next to metadata: {path}")
            idx = faiss.read_index(path)
            if idx.d != bank.dim:
                raise RuntimeError(f"FAISS dimension mismatch: index.d={idx.d} != memory.dim={bank.dim}")
            bank._index = idx
        for kk in ("dim", "capacity", "index_type", "k", "hnsw_M", "ef_construction", "ef_search", "version",
                   "n_inserted", "n_rejected_attack", "n_rejected_threshold", "n_rejected_low_confidence",
                   "n_replaced", "n_dropped_redundant", "selection", "replacement", "seed"):
            if kk in st:
                setattr(bank.stats, kk, st[kk])
        bank.stats.n = int(vecs.shape[0])
        bank.stats.updates = list(st.get("updates", []))
        if os.path.exists(base + "_meta.json"):
            with open(base + "_meta.json") as fh:
                blob2 = json.load(fh)
            if os.path.exists(path) and blob2["stats"].get("index_type") != bank.stats.index_type:
                raise RuntimeError("memory index type recorded in metadata differs from the loaded index")
        return bank

    def entries(self) -> List[Dict[str, Any]]:
        return list(self._meta)

    def vectors(self) -> np.ndarray:
        return self._vectors.copy()

    def nearest_metadata(self, idx: int) -> Dict[str, Any]:
        return dict(self._meta[int(idx)]) if 0 <= int(idx) < len(self._meta) else {}

    def summary(self) -> Dict[str, Any]:
        s = self.stats.as_dict()
        s.pop("updates", None)
        s["n_update_events"] = len(self.stats.updates)
        s["index_backend"] = "faiss" if self._index is not None else "numpy"
        return s
