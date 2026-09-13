"""Member-3 scoring core: memory retrieval, anomaly score, threshold, alert state.

Single scoring definition used by every table, figure and API response (§9):

    z          = L2-normalise(embedding)                    # unit sphere
    s(z)       = max_{m in NN_k(z)}  cos(z, m)              # cosine similarity
    anomaly    = 1 - s(z)          in [0, 2]   (0 = matches a stored normal)

Distance-like language ("anomaly distance", "cosine distance" in the project doc)
always refers to this single quantity; no Euclidean path exists in this module.

Threshold calibration reads ONLY the permitted benign calibration split; the test
split is never an input (asserted in ``calibrate_threshold`` provenance checks and
in ``contract.assert_splits_clean``).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contract import (
    LABEL_ATTACK,
    LABEL_BENIGN,
    SPLIT_CALIBRATION,
    SPLIT_MEMORY,
    SPLIT_ONLINE_BENIGN,
    SPLIT_TEST,
    ContractError,
    l2_normalize,
    mask_graph_sequence,
)
from .memory import AdaptiveMemoryBank

EPS = 1e-12


# --------------------------------------------------------------------------- #
# confidence-aware threshold
# --------------------------------------------------------------------------- #
@dataclass
class ThresholdModel:
    method: str = "confidence_aware"
    value: float = float("nan")
    scale: float = float("nan")          # spread used to normalise the margin (sigma_hat or std)
    median: float = float("nan")
    mad_sigma: float = float("nan")      # 1.4826 * MAD of the calibration scores
    percentile: float = 99.5
    safety_margin: float = 0.25          # in units of scale
    k_mad: float = 3.5
    confidence_sharpness: float = 1.0
    n_calibration: int = 0
    calibrated_on: str = ""
    seed: int = 0
    updated_online: int = 0
    online_history: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yamlable(self) -> Dict[str, Any]:
        return self.as_dict()


def _robust_spread(x: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, 1.4826 * mad


def calibrate_threshold(scores: np.ndarray, cfg: Dict[str, Any], calibrated_on: str,
                        split_tags: Optional[Sequence[str]] = None) -> ThresholdModel:
    """Fit the decision threshold from permitted calibration scores only.

    Methods (ablation-ready, all deterministic):
      fixed_mean3sd      : tau = mean + 3*sd                       (simple baseline)
      fixed_percentile   : tau = percentile(S, p)                  (simple baseline)
      confidence_aware   : tau = max(percentile_p, median + k_mad*MAD_sigma)
                                + safety_margin * MAD_sigma         (project method)

    The MAD fence makes the rule environment-aware: a benign distribution with
    heavy dispersion (busy/noisy network) pushes tau up by a *robust* amount, while
    a tight benign manifold keeps tau near the percentile target - i.e. the
    threshold adapts to traffic characteristics instead of being one global constant.
    """
    tc = dict(cfg.get("threshold", {}))
    method = str(tc.get("strategy", "confidence_aware"))
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size < int(tc.get("min_calibration_samples", 32)):
        raise ContractError(
            f"calibration data missing/too small: n={s.size} < {tc.get('min_calibration_samples', 32)}"
        )
    if np.any(~np.isfinite(s)):
        raise ContractError("calibration scores contain non-finite values")
    if split_tags and SPLIT_TEST in set(split_tags):
        raise PermissionError("THRESHOLD CONTAMINATION BLOCKED: test split offered to calibration")
    p = float(tc.get("percentile", 99.5))
    med, sig = _robust_spread(s)
    pct = float(np.percentile(s, p))
    if method == "fixed_mean3sd":
        tau, scale = float(s.mean() + float(tc.get("fixed_sigma", 3.0)) * s.std(ddof=1)), float(s.std(ddof=1))
    elif method == "fixed_percentile":
        tau, scale = pct, sig if sig > EPS else float(s.std(ddof=1))
    elif method in ("confidence_aware", "adaptive", "confidence-aware"):
        fence = med + float(tc.get("k_mad", 3.5)) * sig
        scale = sig if sig > EPS else float(s.std(ddof=1)) or 1.0
        tau = max(pct, fence) + float(tc.get("safety_margin", 0.25)) * scale
    else:
        raise ContractError(f"unknown threshold.strategy '{method}'")
    if not np.isfinite(tau):
        raise ContractError("threshold calibration produced a non-finite value")
    return ThresholdModel(
        method=method, value=float(tau), scale=float(scale), median=med, mad_sigma=sig, percentile=p,
        safety_margin=float(tc.get("safety_margin", 0.25)), k_mad=float(tc.get("k_mad", 3.5)),
        confidence_sharpness=float(tc.get("confidence_sharpness", 1.0)), n_calibration=int(s.size),
        calibrated_on=calibrated_on, seed=int(cfg.get("seed", 0)),
    )


def confidence_from_score(score: Any, tau: float, scale: float, sharpness: float = 1.0) -> np.ndarray:
    """P(anomalous | score): logistic in the calibration-scaled margin (s - tau)/scale."""
    z = (np.asarray(score, dtype=np.float64) - float(tau)) / (abs(float(scale)) + EPS)
    return 1.0 / (1.0 + np.exp(-float(sharpness) * z))


# --------------------------------------------------------------------------- #
# scorer
# --------------------------------------------------------------------------- #
class Member3Scorer:
    """Member-2 embedding -> memory retrieval -> score -> threshold -> alert state."""

    def __init__(self, adapter: Any, adapter_meta: Dict[str, Any], memory: AdaptiveMemoryBank,
                 threshold: ThresholdModel, cfg: Dict[str, Any],
                 provenance: Optional[Dict[str, str]] = None):
        self.adapter, self.adapter_meta = adapter, dict(adapter_meta or {})
        self.memory, self.threshold, self.cfg = memory, threshold, dict(cfg or {})
        self.k = int(memory.k)
        self.provenance = dict(provenance or {})
        self.normalise_embeddings = not bool(self.adapter_meta.get("l2_normalized", False))
        self.beta = float(threshold.confidence_sharpness)
        self.neighbor_agg = str(cfg.get("retrieval", {}).get("neighbor_aggregation", "max"))

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def build(cls, cfg: Dict[str, Any], adapter: Any, adapter_meta: Dict[str, Any], records: pd.DataFrame,
              memory_labels: Optional[Sequence[str]] = None) -> "Member3Scorer":
        """Fit the normal memory (train split) and calibrate the threshold (calibration split)."""
        dim = int(adapter_meta["embedding_dim"])
        mc = dict(cfg.get("memory", {}))
        emb_dim = int(cfg.get("member2", {}).get("embedding_dim", dim) or dim)
        if emb_dim != dim:
            raise ContractError(f"config embedding_dim {emb_dim} != Member-2 {dim}")
        memory = AdaptiveMemoryBank(
            dim=dim, capacity=int(mc.get("capacity", 4096)), index_type=str(mc.get("index_type", "hnsw")),
            k=int(cfg.get("retrieval", {}).get("k", 8)), hnsw_M=int(mc.get("hnsw_M", 32)),
            ef_construction=int(mc.get("ef_construction", 200)), ef_search=int(mc.get("ef_search", 128)),
            selection=str(mc.get("selection", "kcenter_greedy")), replacement=str(mc.get("replacement", "redundant")),
            seed=int(cfg.get("seed", 0)),
        )
        mem_rows = records[records["split"] == SPLIT_MEMORY]
        if len(mem_rows) == 0:
            raise ContractError(f"no '{SPLIT_MEMORY}' rows: cannot build normal memory bank")
        X_mem = np.vstack(list(mem_rows["embedding"]))
        meta = [{"sample_id": a, "dataset_id": b, "window_id": c, "timestamp": float(d),
                 "source_split": SPLIT_MEMORY}
                for a, b, c, d in zip(mem_rows["sample_id"], mem_rows["dataset_id"], mem_rows["window_id"],
                                      mem_rows["timestamp"])]
        labels = list(mem_rows["label"]) if memory_labels is None else list(memory_labels)
        memory.fit(X_mem, meta, labels)

        cal_rows = records[records["split"] == SPLIT_CALIBRATION]
        if len(cal_rows) == 0:
            raise ContractError(f"no '{SPLIT_CALIBRATION}' rows: refusing to run without permitted calibration data")
        prov = {"memory_source": str(mem_rows["dataset_id"].unique().tolist()),
                "calibration_source": str(cal_rows["dataset_id"].unique().tolist()),
                "test_datasets": str(sorted(records[records["split"] == SPLIT_TEST]["dataset_id"].unique().tolist()))}
        cal_scores, _ = _raw_scores(memory, np.vstack(list(cal_rows["embedding"])), memory.k,
                                     str(cfg.get("retrieval", {}).get("neighbor_aggregation", "max")))
        thr = calibrate_threshold(cal_scores, cfg, calibrated_on=f"{SPLIT_CALIBRATION}@{prov['calibration_source']}",
                                  split_tags=[SPLIT_CALIBRATION])
        scorer = cls(adapter, adapter_meta, memory, thr, cfg, provenance=prov)
        return scorer

    # ---- scoring --------------------------------------------------------- #
    def score_embeddings(self, X: np.ndarray, k: Optional[int] = None,
                         with_neighbours: bool = True) -> Dict[str, Any]:
        Z = l2_normalize(np.asarray(X, dtype=np.float64))
        if Z.ndim == 1:
            Z = Z[None, :]
        sims, idx = self.memory.search(Z, k=k or self.k)
        best = aggregate_sims(sims, self.neighbor_agg)
        score = 1.0 - best
        out: Dict[str, Any] = {"similarity": best, "anomaly_score": score, "neighbor_idx": idx,
                               "neighbor_sims": sims}
        if with_neighbours:
            out["neighbor_ids"] = [
                [self.memory.nearest_metadata(int(j)).get("sample_id") for j in row] for row in idx
            ]
            out["neighbor_datasets"] = [
                [self.memory.nearest_metadata(int(j)).get("dataset_id") for j in row] for row in idx
            ]
        out["confidence"] = confidence_from_score(score, self.threshold.value, self.threshold.scale, self.beta)
        out["is_anomaly"] = score > self.threshold.value
        return out

    def predict(self, records: pd.DataFrame, score_col: str = "anomaly_score") -> pd.DataFrame:
        """Full inference pass over one or more splits; returns the per-sample audit table.

        Evaluation mode is immutable: the memory bank and threshold are frozen for the
        whole pass, so a run cannot leak its own test samples into its own reference set.
        """
        memory_was = self.memory.stats.immutable
        self.memory.set_immutable(True)
        try:
            df = records.copy()
            if "embedding" not in df.columns or df["embedding"].isna().any():
                raise ContractError(
                    "records carry no 'embedding' column: Member 3 does not run the foundation model "
                    "itself. Export embeddings with the Member-2 adapter (contract §B) into the records "
                    "file, or score single windows through the API, which calls adapter.encode() for you.")
            X = np.vstack(list(df["embedding"]))
            res = self.score_embeddings(X)
            df["embedding_dim"] = X.shape[1]
            df["nearest_normal_similarity"] = res["similarity"]
            df[score_col] = res["anomaly_score"]
            df["confidence"] = res["confidence"]
            df["is_anomaly"] = res["is_anomaly"].astype(bool)
            df["threshold"] = float(self.threshold.value)
            df["threshold_method"] = self.threshold.method
            df["memory_version"] = self.memory.stats.version
            df["neighbor_sample_ids"] = [",".join(str(v) for v in lst) for lst in res["neighbor_ids"]]
            df["neighbor_similarities"] = [",".join(f"{v:.6f}" for v in row) for row in res["neighbor_sims"]]
            df["checkpoint_version"] = str(self.adapter_meta.get("checkpoint_version", "unknown"))
            df["dataset_split"] = df["dataset_id"].astype(str) + "/" + df["split"].astype(str)
        finally:
            self.memory.set_immutable(memory_was)
        return df

    # ---- multi-stage alert verification ---------------------------------- #
    def verify_alerts(self, pred: pd.DataFrame, debounce: Optional[int] = None,
                      window_seconds: float = 30.0) -> pd.DataFrame:
        """Consecutive-anomaly confirmation (debounce) + detection delay, per live stream.

        Rule implemented verbatim from the architecture (3 steps by default):
            anomaly streak 1..L-1 -> candidate ; streak L -> CONFIRMED ALERT ;
            any benign window resets the streak to 0.
        """
        L = int(debounce if debounce is not None else self.cfg.get("alerting", {}).get("debounce_steps", 3))
        if L < 1:
            raise ContractError(f"debounce steps must be >= 1, got {L}")
        df = pred.sort_values(["stream_id", "stream_pos"], kind="mergesort").copy()
        if "stream_id" not in df.columns or df["stream_id"].isna().any():
            raise ContractError("records need 'stream_id'/'stream_pos' for temporal alert verification "
                                "(run contract.assign_streams on the dataset)")
        df["consecutive_anomalies"] = 0
        df["alert_state"] = "none"          # none | candidate | confirmed
        df["confirmed_alert"] = False
        df["alert_raised"] = False          # rising edge into "confirmed" = one alert event
        df["episode_detected"] = False
        df["detection_delay_steps"] = np.nan
        df["detection_delay_seconds"] = np.nan
        for stream_key, g in df.groupby("stream_id", sort=False):
            pos = g["stream_pos"].to_numpy(dtype=np.int64)
            ano = g["is_anomaly"].to_numpy(dtype=bool)
            ep = (g["episode_id"].fillna("benign").astype(str).to_numpy() if "episode_id" in g.columns
                  else np.array(["benign"] * len(g), dtype=object))
            streak, first_pos, prev = 0, None, None
            streaks, states = [], []
            confirmed_at: Dict[str, int] = {}
            for i in range(len(pos)):
                if prev is not None and int(pos[i]) != prev + 1:
                    streak, first_pos = 0, None          # gap in the observed stream
                prev = int(pos[i])
                if bool(ano[i]):
                    streak += 1
                    if first_pos is None:
                        first_pos = prev
                    state = "confirmed" if streak >= L else "candidate"
                    if streak >= L:
                        confirmed_at.setdefault(str(ep[i]), prev)
                else:
                    streak, first_pos = 0, None
                    state = "none"
                streaks.append(streak)
                states.append(state)
            df.loc[g.index, "consecutive_anomalies"] = streaks
            df.loc[g.index, "alert_state"] = states
            conf = np.array([s == "confirmed" for s in states], dtype=bool)
            raised = conf.copy()
            raised[1:] = conf[1:] & ~conf[:-1]      # one alert event per confirmed streak
            df.loc[g.index, "confirmed_alert"] = conf
            df.loc[g.index, "alert_raised"] = raised
            for eid, cpos in confirmed_at.items():
                if eid == "benign":
                    continue
                rows = g[g["episode_id"].fillna("benign").astype(str).eq(eid)]
                if len(rows) == 0:
                    continue
                anom_pos = rows.loc[rows["is_anomaly"], "stream_pos"].to_numpy(dtype=np.int64)
                first = int(anom_pos.min()) if anom_pos.size else int(cpos)
                delay = max(0, int(cpos) - first)
                df.loc[rows.index, "episode_detected"] = True
                df.loc[rows.index, "detection_delay_steps"] = float(delay)
                df.loc[rows.index, "detection_delay_seconds"] = float(delay) * float(window_seconds)
        df["anomaly_score"] = df["anomaly_score"].astype(float)
        return df.sort_index()

    # ---- online benign adaptation ---------------------------------------- #
    def adapt_online(self, records: pd.DataFrame, pred: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Concept-drift adaptation of the MEMORY (model weights stay frozen).

        Only samples that are (a) labelled/stated benign by the online feed,
        (b) below the calibrated threshold and (c) confidently benign are inserted.
        """
        ac = dict(self.cfg.get("adaptation", {}))
        if not ac.get("enabled", True):
            return {"status": "disabled", "n_candidates": int(len(records))}
        df = records[records["split"] == SPLIT_ONLINE_BENIGN]
        if len(df) == 0:
            return {"status": "no_online_benign_stream", "n_candidates": 0}
        if pred is None:
            pred = self.predict(df)
        else:
            pred = pred[pred["split"] == SPLIT_ONLINE_BENIGN].copy()
        if str(self.adapter_meta.get("trainable", "false")).lower() == "true":
            raise ContractError("adapter reports a trainable model; online memory adaptation must not retrain")
        self.memory.set_immutable(False)
        actions = []
        try:
            X = np.vstack(list(df["embedding"]))
            # every candidate goes through the memory gate; rejections are logged there, never silently dropped
            for i, (sid, sc, cf, lab) in enumerate(zip(df["sample_id"], pred["anomaly_score"], pred["confidence"],
                                                        df["label"])):
                rec = self.memory.update_if_benign(
                    X[i], confidence=float(cf), score=float(sc), threshold=float(self.threshold.value),
                    metadata={"sample_id": sid, "label": lab, "dataset_id": str(df.iloc[i]["dataset_id"]),
                              "timestamp": float(df.iloc[i]["timestamp"])},
                    allow_online=True, benign_confidence_max=float(ac.get("benign_confidence_max", 0.05)))
                actions.append(str(rec["action"]))
        finally:
            self.memory.set_immutable(True)
        counts = pd.Series(actions).value_counts().to_dict()
        accepted = ("inserted", "redundancy_replaced", "reservoir_replaced")
        return {"status": "ok", "n_candidates": int(len(df)), "actions": counts,
                "n_updates": int(sum(v for k, v in counts.items() if k in accepted)),
                "n_rejected": int(sum(v for k, v in counts.items() if k not in accepted)),
                "memory_size_after": int(self.memory.stats.n), "memory_version_after": int(self.memory.stats.version),
                "model_weights_updated": False}


# --------------------------------------------------------------------------- #
# shared scoring helper (used before the scorer object exists)
# --------------------------------------------------------------------------- #
def aggregate_sims(sims: np.ndarray, mode: str = "max") -> np.ndarray:
    """The one documented neighbour aggregation.  ``max`` = nearest normal neighbour.

    ``mean_of_k`` is provided only for sensitivity checks and is always written into
    the manifest, so a run can never claim the primary definition while using another.
    """
    if mode in ("max", "nearest"):
        return sims.max(axis=1)
    if mode in ("mean_of_k", "mean"):
        return sims.mean(axis=1)
    raise ContractError(f"unknown retrieval.neighbor_aggregation '{mode}' (max | mean_of_k)")


def _raw_scores(memory: AdaptiveMemoryBank, X: np.ndarray, k: int, mode: str = "max") -> Tuple[np.ndarray, np.ndarray]:
    Z = l2_normalize(np.asarray(X, dtype=np.float64))
    if Z.ndim == 1:
        Z = Z[None, :]
    sims, idx = memory.search(Z, k=k)
    return 1.0 - aggregate_sims(sims, mode), sims


# --------------------------------------------------------------------------- #
# latency benchmarking (§28)
# --------------------------------------------------------------------------- #
def _stats_ms(times: Sequence[float]) -> Dict[str, float]:
    t = np.asarray(list(times), dtype=np.float64) * 1000.0
    if t.size == 0:
        return {k: float("nan") for k in ("mean", "median", "p95", "p99", "n")}
    return {"mean": float(t.mean()), "median": float(np.median(t)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "min": float(t.min()), "max": float(t.max()), "n": int(t.size)}


def benchmark_latency(scorer: "Member3Scorer", X: np.ndarray, graphs: Optional[Sequence[Any]] = None,
                      n_samples: int = 200, n_repeat: int = 3, warmup: int = 10) -> Dict[str, Any]:
    """Stage-wise per-sample latency (ms).

    A embedding inference (only if graph sequences are supplied), B FAISS retrieval,
    C thresholding + confidence, D alert verification, E total Member-3 scoring
    (retrieval + scoring + thresholding, i.e. everything Member 3 owns).
    Every number is measured on this machine; nothing here is asserted as a target.
    """
    X = np.asarray(X, dtype=np.float64)
    n = min(int(n_samples), len(X))
    rng = np.random.default_rng(int(scorer.cfg.get("seed", 0)))
    pick = rng.choice(len(X), size=n, replace=False)
    rows = X[pick]
    # stage A is only measurable when one graph sequence per scored window is supplied, aligned
    # with X by position; anything else is reported as NOT RUN rather than guessed or crashed on.
    seqs, a_reason = None, ""
    if graphs is not None:
        if len(graphs) != len(X):
            a_reason = (f"NOT RUN - {len(graphs)} graph sequences supplied for {len(X)} records: stage A needs "
                        "exactly one graph per scored window, in the same order (contract §E)")
        else:
            seqs = [graphs[int(i)] for i in pick]
            if any(s is None for s in seqs):
                raise ContractError("benchmark_latency: graph sequence missing for at least one sampled window")
    acc: Dict[str, List[float]] = {k: [] for k in ("A_embedding", "B_faiss_retrieval", "C_thresholding",
                                                   "D_alert_verification", "E_member3_total")}
    for rep in range(int(n_repeat)):
        if seqs is not None:
            for g in seqs[: max(int(warmup), 0)]:
                scorer.adapter.encode(g)
            for g in seqs:
                t0 = time.perf_counter()
                scorer.adapter.encode(g)
                acc["A_embedding"].append(time.perf_counter() - t0)
        scorer.memory.search(rows[:warmup], k=scorer.k)
        for i in range(n):
            z = rows[i : i + 1]
            t0 = time.perf_counter()
            sims, idx = scorer.memory.search(z, k=scorer.k)
            t1 = time.perf_counter()
            score = float(1.0 - sims.max())
            confidence_from_score([score], scorer.threshold.value, scorer.threshold.scale, scorer.beta)
            t2 = time.perf_counter()
            acc["B_faiss_retrieval"].append(t1 - t0)
            acc["C_thresholding"].append(t2 - t1)
            acc["E_member3_total"].append(t2 - t0)
            mini = pd.DataFrame({"stream_id": ["b"], "stream_pos": [int(i)], "is_anomaly": [score > scorer.threshold.value],
                                 "anomaly_score": [score], "episode_id": ["benign"]})
            t3 = time.perf_counter()
            scorer.verify_alerts(mini, debounce=1)
            acc["D_alert_verification"].append(time.perf_counter() - t3)
    # stage D measured per window (streaming/API cost); also report the amortised batch cost
    mini = pd.DataFrame({"stream_id": ["b"] * n, "stream_pos": np.arange(n),
                         "anomaly_score": scorer.score_embeddings(rows, with_neighbours=False)["anomaly_score"],
                         "episode_id": ["benign"] * n})
    mini["is_anomaly"] = mini["anomaly_score"].to_numpy() > scorer.threshold.value
    mini["threshold"] = scorer.threshold.value
    t0 = time.perf_counter()
    scorer.verify_alerts(mini, debounce=None)
    dt_batch = (time.perf_counter() - t0) / max(n, 1)
    acc["D_alert_verification_batch"] = [dt_batch] * n
    out: Dict[str, Any] = {"per_sample_ms": {k: _stats_ms(v) for k, v in acc.items() if v},
                           "raw_ms": {k: [1000.0 * float(t) for t in v] for k, v in acc.items() if v}}
    out["stages_not_measured"] = [k for k, v in acc.items() if not v]
    if seqs is None:
        out["A_note"] = a_reason or ("NOT RUN - Member 3 was fed cached embeddings; time Member-2 encode() "
                                     "yourself with the adapter from the handoff artefacts")
    out["conditions"] = {
        "n_samples": int(n), "repeats": int(n_repeat), "warmup": int(warmup), "batch_size": 1,
        "device": str(scorer.adapter_meta.get("device", "cpu")), "embedding_dim": int(scorer.memory.stats.dim),
        "k": int(scorer.k), "index_type": str(scorer.memory.stats.index_type),
        "memory_size": int(scorer.memory.stats.n), "state": "warm after per-stage warmup calls",
        "throughput_samples_per_s": (1000.0 / out["per_sample_ms"]["E_member3_total"]["mean"]
                                    if out["per_sample_ms"].get("E_member3_total", {}).get("mean") else float("nan")),
    }
    out["conditions"].update(hardware_info())
    return out


def hardware_info() -> Dict[str, Any]:
    import platform

    info: Dict[str, Any] = {"system": platform.system(), "machine": platform.machine(),
                            "processor": platform.processor(), "cpu_count": 0, "python": platform.python_version()}
    try:
        import os

        info["cpu_count"] = len(os.sched_getaffinity(0))
    except Exception:
        try:
            import multiprocessing as mp

            info["cpu_count"] = mp.cpu_count()
        except Exception:
            pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    info["mem_total_kb"] = int(line.split()[1])
                    break
    except Exception:
        pass
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = "not-installed"
    return info


class AlertTracker:
    """Streaming counterpart of :meth:`Member3Scorer.verify_alerts`.

    Maintains the consecutive-anomaly streak for one live stream and returns the
    same states (none / candidate / confirmed) used offline, so the API cannot
    drift from the evaluation protocol.
    """

    def __init__(self, debounce: int = 3, window_seconds: float = 30.0):
        self.debounce = int(debounce)
        self.window_seconds = float(window_seconds)
        self.streak = 0
        self.first_anomaly_pos: Optional[int] = None
        self.n_confirmed = 0
        self.last_pos: Optional[int] = None
        self.history: List[Dict[str, Any]] = []

    def update(self, score: float, threshold: float, pos: Optional[int] = None) -> Dict[str, Any]:
        anomaly = float(score) > float(threshold)
        if pos is not None and self.last_pos is not None and int(pos) != self.last_pos + 1:
            self.streak, self.first_anomaly_pos = 0, None   # gap in the observed stream (as offline does)
        if pos is not None:
            self.last_pos = int(pos)
        if anomaly:
            self.streak += 1
            if self.first_anomaly_pos is None:
                self.first_anomaly_pos = pos
        else:
            self.streak, self.first_anomaly_pos = 0, None
        confirmed = anomaly and self.streak >= self.debounce
        if confirmed:
            self.n_confirmed += 1
        delay = None
        if confirmed and self.first_anomaly_pos is not None and pos is not None:
            delay = max(0, int(pos) - int(self.first_anomaly_pos))
        state = {"anomaly_score": float(score), "threshold": float(threshold), "is_anomaly": bool(anomaly),
                 "consecutive_anomalies": int(self.streak), "required_consecutive": self.debounce,
                 "alert_state": "confirmed" if confirmed else ("candidate" if anomaly else "none"),
                 "confirmed_alert": bool(confirmed), "detection_delay_steps": delay,
                 "detection_delay_seconds": (delay * self.window_seconds) if delay is not None else None,
                 "pending_memory_update": bool(not anomaly), "n_confirmed_alerts": self.n_confirmed}
        self.history.append(state)
        return state
