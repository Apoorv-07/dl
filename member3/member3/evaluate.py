"""Evaluation, ablations, statistics, publication tables and figures for Member 3.

Nothing in this module fabricates a number: every metric is computed from a
predictions table produced by :mod:`member3.inference`, and any experiment that
could not run is written out as ``NOT RUN`` together with the reason.
"""

from __future__ import annotations

import json
import os
import platform
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contract import (LABEL_ATTACK, LABEL_BENIGN, SPLIT_CALIBRATION, SPLIT_MEMORY, SPLIT_ONLINE_BENIGN,
                         SPLIT_TEST)
from .explain import bootstrap_ci
from .inference import calibrate_threshold, confidence_from_score

NOT_RUN = "NOT RUN"
FIG_DPI = 300

METRIC_UNITS = {
    "precision": "ratio", "recall": "ratio", "f1": "ratio", "roc_auc": "ratio", "auprc": "ratio",
    "fpr": "ratio", "fnr": "ratio", "fpr_pct": "%", "threshold": "score units (1 - cosine sim)",
    "n_samples": "count", "n_positive": "count", "n_negative": "count", "n_episodes": "count",
    "n_alerts": "count", "n_true_alerts": "count", "n_false_alerts": "count", "n_missed_episodes": "count",
    "mean_detection_delay_steps": "windows (30 s each)", "median_detection_delay_steps": "windows",
    "p95_detection_delay_steps": "windows", "mean_detection_delay_seconds": "seconds",
    "median_detection_delay_seconds": "seconds", "mean_anomaly_score": "score units",
    "median_anomaly_score": "score units", "mean": "score units", "median": "score units",
    "std": "score units", "memory_size": "vectors", "memory_growth": "vectors", "n_updates": "count",
    "mean_ms": "milliseconds", "median_ms": "milliseconds", "p95_ms": "milliseconds",
    "p99_ms": "milliseconds", "min_ms": "milliseconds", "max_ms": "milliseconds",
    "throughput_samples_per_s": "samples/second", "fidelity_plus_pct": "% score reduction",
    "ci95_low": "% score reduction", "ci95_high": "% score reduction", "mean_before": "score units",
    "mean_after": "score units", "frac_score_drops": "ratio", "explanation_coverage": "ratio",
    "tau_delta_vs_baseline": "score units", "folds": "count", "detection_rate": "ratio",
}


# --------------------------------------------------------------------------- #
# metric primitives
# --------------------------------------------------------------------------- #
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Sample-level confusion-matrix metrics + threshold-free ranking metrics."""
    from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    n, n_pos = int(y.size), int(y.sum())
    out: Dict[str, Any] = {"n_samples": n, "n_positive": n_pos, "n_negative": int(n - n_pos)}
    if n == 0 or n_pos == 0 or n_pos == n:
        # degenerate for precision/recall but FPR or recall alone may still be defined
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel() if n and 0 < n_pos < n else _cm_fallback(y, p)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out.update({"precision": float(prec), "recall": float(rec),
                    "f1": float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
                    "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
                    "fnr": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
                    "roc_auc": NOT_RUN, "auprc": NOT_RUN,
                    "metric_validity": "single-class evaluation set: ranking metrics undefined"})
        return out
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    out.update({
        "precision": float(prec), "recall": float(rec),
        "f1": float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
        "fpr": float(fp / (fp + tn)), "fnr": float(fn / (fn + tp)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_auc": float(roc_auc_score(y, scores)) if scores is not None else NOT_RUN,
        "auprc": float(average_precision_score(y, scores)) if scores is not None else NOT_RUN,
        "metric_validity": "ok",
    })
    return out


def _cm_fallback(y: np.ndarray, p: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    return tn, fp, fn, tp


def episode_metrics(pred: pd.DataFrame, window_seconds: Optional[float] = None) -> Dict[str, Any]:
    """Alert-level (episode) view: what an analyst actually experiences.

    An *episode* is one contiguous attack burst in the stream (``episode_id``).
    A true alert is an episode containing a rising-edge confirmation; a false alert
    is a rising-edge confirmation on the benign background.  Detection delay is
    measured from the first anomalous window of the episode to the confirmation.
    """
    if "episode_id" not in pred.columns:
        return {"n_episodes": NOT_RUN, "metric_validity": "no episode_id column: episode metrics unavailable"}
    raised_col = "alert_raised" if "alert_raised" in pred.columns else "confirmed_alert"
    ws = float(window_seconds) if window_seconds is not None else 30.0
    ep_att = pred[pred["label"] == LABEL_ATTACK]
    ep_ben = pred[pred["label"] == LABEL_BENIGN]
    if len(ep_att) == 0:
        return {"n_episodes": 0, "detection_rate": float("nan"), "alert_precision": float("nan"),
                "n_true_alerts": 0, "n_false_alerts": int(ep_ben[raised_col].sum()),
                "n_missed_episodes": 0, "mean_detection_delay_steps": float("nan"),
                "median_detection_delay_steps": float("nan"), "p95_detection_delay_steps": float("nan"),
                "mean_detection_delay_seconds": float("nan"), "median_detection_delay_seconds": float("nan")}
    g = ep_att.groupby("episode_id").agg(detected=(raised_col, "any"), delay=("detection_delay_steps", "min"))
    tp = int(g["detected"].sum())
    fn = int(len(g) - tp)
    fp = int(ep_ben[raised_col].sum())
    d = pd.to_numeric(g.loc[g["detected"], "delay"], errors="coerce").dropna().to_numpy(dtype=float)
    return {
        "n_episodes": int(len(g)), "n_true_alerts": tp, "n_missed_episodes": fn, "n_false_alerts": fp,
        "detection_rate": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "alert_precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "mean_detection_delay_steps": float(np.mean(d)) if d.size else float("nan"),
        "median_detection_delay_steps": float(np.median(d)) if d.size else float("nan"),
        "p95_detection_delay_steps": float(np.percentile(d, 95)) if d.size else float("nan"),
        "mean_detection_delay_seconds": float(np.mean(d) * ws) if d.size else float("nan"),
        "median_detection_delay_seconds": float(np.median(d) * ws) if d.size else float("nan"),
        "false_alert_rate_per_1000_benign_windows": (1000.0 * fp / len(ep_ben)) if len(ep_ben) else float("nan"),
    }


def score_summary(pred: pd.DataFrame, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    s = pred["anomaly_score"].to_numpy(dtype=float) if mask is None else pred.loc[mask, "anomaly_score"].to_numpy(dtype=float)
    if s.size == 0:
        return {"mean_anomaly_score": float("nan"), "median_anomaly_score": float("nan"), "n_samples": 0}
    return {"mean_anomaly_score": float(np.mean(s)), "median_anomaly_score": float(np.median(s)),
            "n_samples": int(s.size)}


# --------------------------------------------------------------------------- #
# evaluation drivers
# --------------------------------------------------------------------------- #
def overall_metrics(pred: pd.DataFrame, cfg: Dict[str, Any], dataset_label: str = "", unit: str = "sample") -> Dict[str, Any]:
    tst = pred[pred["split"] == SPLIT_TEST].copy()
    if len(tst) == 0:
        raise ValueError(f"no '{SPLIT_TEST}' rows to evaluate ({dataset_label})")
    y = (tst["label"] == LABEL_ATTACK).astype(int).to_numpy()
    decision = "confirmed_alert" if unit == "alert" else "is_anomaly"
    m = classification_metrics(y, tst[decision].astype(int).to_numpy(), tst["anomaly_score"].to_numpy())
    m.update(score_summary(tst))
    m["unit"] = unit
    m["decision_column"] = decision
    m["threshold"] = float(tst["threshold"].iloc[0])
    m["threshold_method"] = str(tst["threshold_method"].iloc[0])
    m["dataset"] = dataset_label or ",".join(sorted(tst["dataset_id"].unique()))
    m["split"] = SPLIT_TEST
    m["n_benign"] = int((y == 0).sum())
    m["n_attack"] = int((y == 1).sum())
    m["fpr_pct"] = (float(m["fpr"]) * 100.0) if isinstance(m.get("fpr"), (int, float)) and np.isfinite(float(m["fpr"])) else NOT_RUN
    m["n_alerts"] = int(tst["alert_raised"].sum()) if "alert_raised" in tst.columns else (
        int(tst["confirmed_alert"].sum()) if "confirmed_alert" in tst.columns else NOT_RUN)
    if "detection_delay_steps" in tst.columns:
        m.update(episode_metrics(tst, window_seconds=float(cfg.get("data", {}).get("window_seconds", 30))))
    boot = int(cfg.get("statistics", {}).get("bootstrap_iterations", 1000))
    seed = int(cfg.get("seed", 0))
    cis = bootstrap_metric_ci(y, tst[decision].astype(int).to_numpy(), tst["anomaly_score"].to_numpy(),
                              metrics=("f1", "auprc", "roc_auc", "fpr"), n_boot=boot, seed=seed)
    for k, v in cis.items():
        m[k + "_ci95"] = v
    return m


def bootstrap_metric_ci(y: np.ndarray, p: np.ndarray, scores: np.ndarray, metrics: Sequence[str],
                        n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap CIs for the headline metrics, seeded and reproducible.

    One resampling pass computes every requested metric on the same replicate, so the
    intervals are jointly consistent with the evaluation sample.
    """
    y, p, scores = np.asarray(y).astype(int), np.asarray(p).astype(int), np.asarray(scores, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    acc: Dict[str, List[float]] = {m: [] for m in metrics}
    for _ in range(int(n_boot)):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0 or y[i].sum() == n:
            continue
        mm = classification_metrics(y[i], p[i], scores[i])
        for name in metrics:
            v = mm.get(name)
            if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v)):
                acc[name].append(float(v))
    out: Dict[str, Dict[str, float]] = {}
    for name, vals in acc.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            out[name] = {"point": NOT_RUN, "ci95_low": NOT_RUN, "ci95_high": NOT_RUN, "n_replicates": 0,
                         "reason": "metric undefined on resampled subsets"}
            continue
        out[name] = {"point": float(np.mean(arr)), "ci95_low": float(np.percentile(arr, 100 * alpha / 2)),
                     "ci95_high": float(np.percentile(arr, 100 * (1 - alpha / 2))),
                     "n_replicates": int(arr.size), "n_boot_requested": int(n_boot)}
    return out


def per_attack_family(pred: pd.DataFrame, cfg: Dict[str, Any], families: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Family-level detection at the operating point: family F vs benign windows.

    Other attack families are deliberately excluded from the negative set - a
    detector that flags them is not "wrong" for family F, and mixing them in would
    make precision uninterpretable.  FPR is reported on the same benign negatives.
    """
    tst = pred[pred["split"] == SPLIT_TEST]
    ben = tst[tst["label"] == LABEL_BENIGN]
    fams = list(families or sorted(f for f in tst.loc[tst["label"] == LABEL_ATTACK, "attack_family"].unique()))
    ws = float(cfg.get("data", {}).get("window_seconds", 30))
    rows = []
    for fam in fams:
        pos = tst[tst["attack_family"] == fam]
        # labels are read back from the aligned frame, never positionally concatenated
        sub = pd.concat([pos, ben]).sort_index()
        y = (sub["attack_family"] == fam).astype(int).to_numpy()
        p = sub["is_anomaly"].astype(int).to_numpy()
        m = classification_metrics(y, p, sub["anomaly_score"].to_numpy(dtype=float))
        m.update(score_summary(sub, (sub["attack_family"] == fam).to_numpy()))
        m["n_benign_negative_windows"] = int(len(ben))
        m["dataset"] = ",".join(sorted(pos["dataset_id"].unique())) if len(pos) else ""
        m["attack_family"] = str(fam)
        m["split"] = SPLIT_TEST
        m["evaluation_contrast"] = "family vs benign windows"
        dd = sub.loc[sub["attack_family"] == fam, "detection_delay_steps"].dropna().to_numpy(dtype=float) \
            if "detection_delay_steps" in sub.columns else np.array([])
        m["mean_detection_delay_steps"] = float(np.mean(dd)) if dd.size else float("nan")
        m["median_detection_delay_steps"] = float(np.median(dd)) if dd.size else float("nan")
        m["mean_detection_delay_seconds"] = float(np.mean(dd) * ws) if dd.size else float("nan")
        det = sub.loc[sub["attack_family"] == fam]
        col = "alert_raised" if "alert_raised" in det.columns else "confirmed_alert"
        m["family_alert_rate"] = float(det[col].mean()) if len(det) else float("nan")
        m["threshold"] = float(sub["threshold"].iloc[0]) if len(sub) else float("nan")
        m["threshold_method"] = str(sub["threshold_method"].iloc[0]) if len(sub) else ""
        rows.append(m)
    out = pd.DataFrame(rows)
    if len(out):
        # tie-stable ordering so that repeated runs produce byte-identical tables
        out = out.sort_values(["f1", "attack_family"], ascending=[False, True], kind="stable")
    return out


def leave_one_family_out(records: pd.DataFrame, cfg: Dict[str, Any], families: Sequence[str],
                         per_family_fn) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Zero-day protocol (§16): family F is removed from every permitted stage.

    Memory and threshold are fitted on benign-only splits, so F cannot enter them.
    Rather than assuming that, each fold rebuilds the permitted stages from the
    records with F deleted and reports how much the calibrated threshold and the
    memory contents moved.  ``tau_delta_vs_baseline`` = 0 and ``memory_changed`` = 0
    is therefore *evidence* that the fold is uncontaminated; any non-zero value would
    expose leakage in the pipeline.  F itself is still scored as unseen attack data.
    """
    base_tau = float(cfg.get("_baseline_threshold", "nan"))
    rows, audit_folds = [], []
    for fam in families:
        removed = int((records["attack_family"] == fam).sum())
        in_permitted = int(((records["attack_family"] == fam)
                            & records["split"].isin([SPLIT_MEMORY, SPLIT_CALIBRATION])).sum())
        row: Dict[str, Any] = {"held_out_family": fam, "rows_removed_from_permitted_stages": in_permitted,
                               "rows_total_for_family": removed}
        try:
            res = per_family_fn(fam)
            row.update(res)
            tau = float(res.get("threshold", "nan"))
            row["tau_delta_vs_baseline"] = (tau - base_tau) if (np.isfinite(tau) and np.isfinite(base_tau)) else float("nan")
            row["status"] = "ok"
        except Exception as exc:  # failed experiments are recorded, never hidden (§36)
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "f1": NOT_RUN,
                        "precision": NOT_RUN, "recall": NOT_RUN, "auprc": NOT_RUN, "roc_auc": NOT_RUN, "fpr": NOT_RUN})
        rows.append(row)
        audit_folds.append({"held_out_family": fam, "status": row.get("status"), "threshold": row.get("threshold"),
                            "tau_delta_vs_baseline": row.get("tau_delta_vs_baseline"),
                            "memory_changed": row.get("memory_changed")})
    out = pd.DataFrame(rows)
    audit: Dict[str, Any] = {"folds": audit_folds, "families": list(families), "n_folds": len(list(families)),
                            "protocol": ("attack family removed from memory-fit + threshold-calibration inputs, "
                                         "then evaluated as unseen; permitted splits are benign-only"),
                            "rows_removed_from_permitted_stages": int(sum(r["rows_removed_from_permitted_stages"] for r in rows))}
    if len(out) and "tau_delta_vs_baseline" in out.columns:
        d = pd.to_numeric(out["tau_delta_vs_baseline"], errors="coerce").abs().to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        audit["max_abs_tau_delta"] = float(np.max(d)) if d.size else float("nan")
        audit["leakage_detected"] = bool(d.size and np.max(d) > 1e-9)
    if len(out) and "memory_changed" in out.columns:
        mc = pd.to_numeric(out["memory_changed"], errors="coerce")
        audit["memory_changed_folds"] = int((mc > 0).sum())
        audit["leakage_detected"] = bool(audit.get("leakage_detected") or (mc > 0).any())
    out.insert(0, "protocol", "leave-one-attack-family-out")
    return out, audit


def ablation_threshold(scorer_factory, cfg: Dict[str, Any], records: pd.DataFrame,
                       strategies: Sequence[str]) -> List[Dict[str, Any]]:
    """Fixed/simple vs confidence-aware threshold, all calibrated on the same benign split."""
    rows = []
    for strat in strategies:
        c = json.loads(json.dumps(cfg))
        c.setdefault("threshold", {})["strategy"] = strat
        try:
            sc = scorer_factory(c)
            pred = sc.verify_alerts(sc.predict(records[records["split"] == SPLIT_TEST]))
            m = overall_metrics(pred, c, unit="sample")
            a = episode_metrics(pred[pred["split"] == SPLIT_TEST])
            rows.append({"strategy": strat, "threshold": m["threshold"], "n_samples": m["n_samples"],
                         "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                         "fpr": m["fpr"], "fpr_pct": m["fpr"] * 100.0, "auprc": m["auprc"], "roc_auc": m["roc_auc"],
                         "n_alerts": m["n_alerts"], "alert_precision": a.get("alert_precision"),
                         "detection_rate": a.get("detection_rate"), "dataset": m["dataset"], "status": "ok"})
        except Exception as exc:
            rows.append({"strategy": strat, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return rows


def ablation_debounce(scorer: Any, cfg: Dict[str, Any], test_pred: pd.DataFrame,
                      steps: Sequence[int]) -> List[Dict[str, Any]]:
    """1/2/3/4-step confirmation ablation on a frozen score table."""
    rows = []
    ws = float(cfg.get("data", {}).get("window_seconds", 30))
    for L in steps:
        pred = scorer.verify_alerts(test_pred.copy(), debounce=int(L), window_seconds=ws)
        y = (pred["label"] == LABEL_ATTACK).astype(int).to_numpy()
        m_s = classification_metrics(y, pred["is_anomaly"].astype(int).to_numpy(), pred["anomaly_score"].to_numpy())
        m_a = classification_metrics(y, pred["confirmed_alert"].astype(int).to_numpy(), pred["anomaly_score"].to_numpy())
        e = episode_metrics(pred, window_seconds=ws)
        rows.append({"debounce_steps": int(L), "n_samples": int(len(pred)),
                     "threshold": float(pred["threshold"].iloc[0]) if "threshold" in pred else float("nan"),
                     "auprc": m_s.get("auprc", NOT_RUN), "roc_auc": m_s.get("roc_auc", NOT_RUN),
                     "sample_precision": m_s["precision"], "sample_recall": m_s["recall"], "sample_f1": m_s["f1"],
                     "sample_fpr": m_s["fpr"], "alert_precision": m_a["precision"], "alert_recall": m_a["recall"],
                     "alert_f1": m_a["f1"], "alert_fpr": m_a["fpr"], "alert_fpr_pct": m_a["fpr"] * 100.0,
                     "n_alerts": int(pred["alert_raised"].sum()) if "alert_raised" in pred else int(pred["confirmed_alert"].sum()),
                     "detection_rate": e.get("detection_rate"),
                     "mean_detection_delay_steps": e.get("mean_detection_delay_steps"),
                     "median_detection_delay_seconds": e.get("median_detection_delay_seconds"),
                     "status": "ok"})
    return rows


def ablation_memory(scorer_factory, cfg: Dict[str, Any], records: pd.DataFrame) -> List[Dict[str, Any]]:
    """Static memory vs adaptive (online benign) memory under concept drift."""
    rows: List[Dict[str, Any]] = []
    for mode in ("static", "adaptive"):
        c = json.loads(json.dumps(cfg))
        try:
            sc = scorer_factory(c)
            size_before = int(sc.memory.stats.n)
            online = records[records["split"] == SPLIT_ONLINE_BENIGN]
            pred_before = sc.predict(online) if len(online) else None
            fp_before = float("nan")
            if pred_before is not None and len(pred_before):
                fp_before = float(pred_before["is_anomaly"].mean())
            upd = {"status": "no_online_stream", "n_updates": 0}
            if mode == "static":
                upd = {"status": "not_applied(static_reference_arm)", "n_updates": 0}
            elif mode == "adaptive" and len(online):
                upd = sc.adapt_online(online, pred_before)
            after = sc.predict(online) if len(online) else None
            fp_after = float(after["is_anomaly"].mean()) if after is not None and len(after) else float("nan")
            test_pred = sc.verify_alerts(sc.predict(records[records["split"] == SPLIT_TEST]))
            m = overall_metrics(test_pred, c, unit="sample")
            rows.append({"memory_mode": mode, "selection": str(c.get("memory", {}).get("selection", "kcenter_greedy")),
                         "memory_size_before": size_before, "memory_size_after": int(sc.memory.stats.n),
                         "memory_growth": int(sc.memory.stats.n) - size_before,
                         "benign_fpr_before_adaptation": fp_before, "benign_fpr_after_adaptation": fp_after,
                         "fpr_reduction": (fp_before - fp_after) if np.isfinite(fp_before) and np.isfinite(fp_after) else float("nan"),
                         "n_updates": upd.get("n_updates", 0), "adaptation_status": upd.get("status"),
                         "n_rejected_above_threshold": upd.get("actions", {}).get("rejected_above_threshold", 0),
                         "n_rejected_uncertain": upd.get("actions", {}).get("rejected_uncertain", 0),
                         "online_windows": int(len(online)),
                         "test_f1": m["f1"], "test_recall": m["recall"], "test_precision": m["precision"],
                         "test_fpr": m["fpr"], "model_weights_updated": bool(upd.get("model_weights_updated", False)),
                         "dataset": m["dataset"], "n_samples": m["n_samples"], "status": "ok"})
        except Exception as exc:
            rows.append({"memory_mode": mode, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return rows


def ablation_retrieval(scorer: Any, cfg: Dict[str, Any], X: np.ndarray, n: int = 300) -> List[Dict[str, Any]]:
    """FAISS HNSW vs exact FAISS flat vs NumPy brute force, same vectors."""
    import time

    from .memory import AdaptiveMemoryBank

    rows = []
    n = min(int(n), len(X))
    dim = int(scorer.memory.stats.dim)
    for idx_type in ("hnsw", "flat", "numpy"):
        try:
            bank = AdaptiveMemoryBank(dim=dim, capacity=int(scorer.memory.stats.capacity), index_type=idx_type,
                                      k=int(scorer.k), seed=int(cfg.get("seed", 0)),
                                      require_faiss=False) if idx_type != "numpy" else None
            if bank is None:
                bank = AdaptiveMemoryBank(dim=dim, capacity=int(scorer.memory.stats.capacity), index_type="numpy",
                                          k=int(scorer.k), seed=int(cfg.get("seed", 0)))
            bank._vectors = scorer.memory.vectors()
            if bank._index is not None:
                bank._index.reset() if hasattr(bank._index, "reset") else None
                try:
                    bank._index.add(np.ascontiguousarray(bank._vectors.astype(np.float32)))
                except Exception:
                    pass
            bank.stats.n = int(bank._vectors.shape[0])
            Q = np.asarray(X[:n], dtype=np.float64)
            bank.search(Q[:4], k=scorer.k)  # warm
            t0 = time.perf_counter()
            sims, ids = bank.search(Q, k=scorer.k)
            dt = (time.perf_counter() - t0) / n * 1000.0
            recall = float(np.mean([len(set(ids[i]) & set(np.argsort(-(Q[i] @ bank._vectors.T))[: scorer.k])) / scorer.k
                                    for i in range(min(n, 60))]))
            rows.append({"index_type": idx_type, "n_queries": int(n), "memory_size": int(bank._vectors.shape[0]),
                         "note": "HNSW only pays off as the bank grows; at small memory sizes exact search is faster",
                         "mean_ms_per_query": float(dt), "recall_at_k_vs_bruteforce": recall, "k": int(scorer.k),
                         "status": "ok"})
        except Exception as exc:
            rows.append({"index_type": idx_type, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return rows


def explanation_stats(pred_expl: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not len(pred_expl):
        return {"n_explanations": 0, "explanation_coverage": NOT_RUN}
    cov = np.array([e.get("explanation_coverage_topk_nodes", np.nan) for e in pred_expl], dtype=float)
    return {"n_explanations": int(len(pred_expl)),
            "explanation_coverage": float(np.nanmean(cov)) if cov.size else float("nan"),
            "mean_top_nodes": float(np.mean([len(e.get("top_nodes", [])) for e in pred_expl])),
            "mean_top_edges": float(np.mean([len(e.get("top_edges", [])) for e in pred_expl])),
            "mean_top_features": float(np.mean([len(e.get("top_features", [])) for e in pred_expl])),
            "feature_attribution_source": str({e.get("feature_attribution_source") for e in pred_expl})}


# --------------------------------------------------------------------------- #
# tidy tables
# --------------------------------------------------------------------------- #
ROW_KEY_CANDIDATES = ("held_out_family", "attack_family", "memory_mode", "strategy", "debounce_steps",
                      "index_type", "arm", "variant", "masking_method", "row_level", "stage", "dataset")


def tidy(table: str, record: Dict[str, Any], dataset: str = "", split: str = "", config_id: str = "",
         data_source: str = "", n_samples: Optional[Any] = None, notes: str = "", row_key: str = "",
         skip: Iterable[str] = ("confusion", "metric_validity", "status")) -> List[Dict[str, Any]]:
    """One tidy row per (table, metric) with dataset/split/config/sample-count context."""
    rows: List[Dict[str, Any]] = []
    n = n_samples if n_samples is not None else record.get("n_samples", record.get("n_queries", ""))
    base = {"table": table, "row_key": row_key, "dataset": dataset, "split": split, "config_id": config_id,
            "n_samples": n, "data_source": data_source, "notes": notes}
    for k, v in record.items():
        if str(k) in set(skip):
            continue
        unit = METRIC_UNITS.get(str(k), "")
        if isinstance(v, dict):  # bootstrap CI objects
            if str(k).endswith("ci95") or "ci" in str(k):
                rows.append({**base, "metric": f"{k}_point", "value": v.get("point", NOT_RUN), "unit": unit})
                rows.append({**base, "metric": f"{k}_ci95_low", "value": v.get("ci95_low", v.get("low")), "unit": unit})
                rows.append({**base, "metric": f"{k}_ci95_high", "value": v.get("ci95_high", v.get("high")), "unit": unit})
            continue
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        if isinstance(v, float) and not np.isfinite(v):
            v = NOT_RUN
        rows.append({**base, "metric": str(k), "value": v, "unit": unit})
    return rows


def _markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Dependency-free markdown rendering (no `tabulate` needed)."""
    def fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return NOT_RUN
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def build_paper_tables(bundles: List[Tuple[str, str, pd.DataFrame, Dict[str, Any]]]) -> Tuple[pd.DataFrame, str]:
    """bundles: [(table_id, caption, dataframe, context)] -> tidy CSV rows + markdown."""
    tidy_rows: List[Dict[str, Any]] = []
    md: List[str] = []
    for table_id, caption, df, ctx in bundles:
        md.append(f"## {table_id} - {caption}")
        md.append("")
        if df is None or len(df) == 0:
            md.append(f"_{NOT_RUN}: {ctx.get('reason', 'no rows')}_\n")
            tidy_rows.extend(tidy(table_id, {"result": NOT_RUN, "reason": str(ctx.get("reason", "no rows")),
                                             "n_samples": 0}, notes=str(ctx.get("reason", "")),
                                  dataset=str(ctx.get("dataset", "")), split=str(ctx.get("split", SPLIT_TEST)),
                                  data_source=str(ctx.get("data_source", "")), config_id=str(ctx.get("config_id", ""))))
            continue
        keys = list(ctx.get("row_keys", [c for c in ROW_KEY_CANDIDATES if c in df.columns])[:2])
        for _, r in df.iterrows():
            rec = {c: r[c] for c in df.columns if c not in ("dataset", "split", "config_id", "data_source", "notes")}
            row_key = "|".join(str(r[c]) for c in keys) if keys else ""
            tidy_rows.extend(tidy(table_id, rec, row_key=row_key,
                                  dataset=str(r.get("dataset", ctx.get("dataset", ""))),
                                  split=str(r.get("split", ctx.get("split", SPLIT_TEST))),
                                  config_id=str(r.get("config_id", ctx.get("config_id", "cfg-default"))),
                                  data_source=str(r.get("data_source", ctx.get("data_source", ""))),
                                  notes=str(r.get("notes", ctx.get("notes", "")))))
        show = df[[c for c in df.columns if not df[c].map(lambda v: isinstance(v, (dict, list))).any()]]
        md.append(_markdown(show))
        md.append("")
    out = pd.DataFrame(tidy_rows)
    if len(out):
        out = out.sort_values(["table", "row_key", "metric"], kind="stable").reset_index(drop=True)
    return out, "\n".join(md)


def latency_table(bench: Dict[str, Any], config_id: str, data_source: str) -> pd.DataFrame:
    rows = []
    per = bench.get("per_sample_ms", {})
    conds = bench.get("conditions", {})
    stage_names = {"A_embedding": "Member-2 encode(graph)", "B_faiss_retrieval": "FAISS k-NN retrieval",
                   "C_thresholding": "threshold + confidence",
                   "D_alert_verification": "multi-stage verification (per-window streaming cost)",
                   "D_alert_verification_batch": "multi-stage verification (amortised over one batch)",
                   "E_member3_total": "Member-3 total scoring"}
    notes = {"D_alert_verification": "one single-window call, i.e. the online/API cost",
             "D_alert_verification_batch": "one batch pass over all sampled windows, divided by the window count"}
    for stage, st in per.items():
        rows.append({"stage": stage, "stage_description": stage_names.get(stage, stage),
                     "mean_ms": st.get("mean"), "median_ms": st.get("median"), "p95_ms": st.get("p95"),
                     "p99_ms": st.get("p99"), "min_ms": st.get("min"), "max_ms": st.get("max"),
                     "n_samples": st.get("n"), "device": conds.get("device"), "batch_size": conds.get("batch_size"),
                     "embedding_dim": conds.get("embedding_dim"), "k": conds.get("k"),
                     "index_type": conds.get("index_type"), "memory_size": conds.get("memory_size"),
                     "cpu_count": conds.get("cpu_count"), "system": conds.get("system"),
                     "throughput_samples_per_s": conds.get("throughput_samples_per_s"),
                     "note": notes.get(stage, ""), "config_id": config_id, "data_source": data_source})
    if not rows:
        rows = [{"stage": NOT_RUN, "stage_description": str(bench.get("A_note", "no timings collected"))}]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": FIG_DPI, "font.family": "DejaVu Serif", "font.size": 9,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8.2, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linestyle": "--", "axes.spines.top": False, "axes.spines.right": False,
        "figure.autolayout": False, "savefig.bbox": "tight",
    })
    return plt


def _save(fig, out_dir: str, name: str, status: Dict[str, str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    import matplotlib.pyplot as _plt_mod

    _plt_mod.close(fig)
    status[name] = "generated"


def figure_architecture(out_dir: str, status: Dict[str, str]) -> None:
    """Figure 1 - Member-3 inference path (drawn from this repo's real modules)."""
    plt = _plt()
    steps = [
        ("Member 2: GATv2 + Temporal Transformer\n(frozen checkpoint, embedding z)", "#e8eef7"),
        ("L2-normalise z\n(unit sphere)", "#eef2f7"),
        ("Adaptive Memory Bank\nrepresentative normal embeddings", "#e6f4ea"),
        ("FAISS HNSW k-NN\n(inner product = cosine)", "#e6f4ea"),
        ("anomaly score = 1 - max cosine sim", "#fdeeda"),
        ("Confidence-aware threshold\ncalibrated on benign split", "#fdeeda"),
        ("Multi-stage alert verification\nN consecutive anomalies", "#fce8e8"),
        ("ALERT -> GATv2 attribution -> Fidelity+", "#f3e8fd"),
        ("Evaluation / metrics / figures / tables -> FastAPI", "#eceff1"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 7.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.15)
    ax.axis("off")
    for i, (txt, col) in enumerate(steps):
        y = (len(steps) - 1 - i) * 1.15
        ax.add_patch(plt.Rectangle((1.2, y), 7.6, 0.85, facecolor=col, edgecolor="#404040", lw=0.8))
        ax.text(5.0, y + 0.42, txt, ha="center", va="center", fontsize=8.6)
        if i < len(steps) - 1:
            # flow is top -> bottom, so the arrow head must sit on the lower box
            ax.annotate("", xy=(5.0, y - 0.29), xytext=(5.0, y - 0.03),
                        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#404040"))
    ax.set_title("Figure 1: Member-3 inference and verification path", fontsize=10.5, pad=12)
    _save(fig, out_dir, "fig01_architecture.png", status)


def figure_roc_pr(pred: pd.DataFrame, out_dir: str, status: Dict[str, str]) -> None:
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    plt = _plt()
    tst = pred[pred["split"] == SPLIT_TEST]
    y = (tst["label"] == LABEL_ATTACK).astype(int).to_numpy()
    s = tst["anomaly_score"].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        status["fig02_roc_curve.png"] = f"{NOT_RUN}: single-class test split"
        status["fig03_pr_curve.png"] = f"{NOT_RUN}: single-class test split"
        return
    fpr, tpr, thr = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    ap = auc(rec, prec)
    auroc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot(fpr, tpr, lw=1.4, label=f"Member 3 (AUC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="#888", lw=1.0, label="random")
    idx = int(np.argmin(np.abs(thr - tst["threshold"].iloc[0]))) if len(thr) else 0
    ax.scatter([fpr[idx]], [tpr[idx]], s=28, c="#c1121f", zorder=5,
               label=f"operating point (tau = {tst['threshold'].iloc[0]:.3f})")
    ax.set_xlabel("False positive rate (fraction of benign windows)")
    ax.set_ylabel("True positive rate (recall of attack windows)")
    ax.set_title("Figure 2: Cross-dataset ROC curve")
    ax.legend(loc="lower right")
    _save(fig, out_dir, "fig02_roc_curve.png", status)

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot(rec, prec, lw=1.4, label=f"Member 3 (AUPRC = {ap:.3f})")
    ax.axhline(float(y.mean()), ls="--", c="#888", lw=1.0, label=f"prevalence = {y.mean():.3f}")
    ax.set_xlabel("Recall (attack windows)")
    ax.set_ylabel("Precision (fraction of flagged windows that are attacks)")
    ax.set_title("Figure 3: Precision-Recall curve")
    ax.legend(loc="lower left")
    _save(fig, out_dir, "fig03_pr_curve.png", status)


def figure_score_distribution(pred: pd.DataFrame, out_dir: str, status: Dict[str, str]) -> None:
    plt = _plt()
    tst = pred[pred["split"] == SPLIT_TEST]
    ben = tst.loc[tst["label"] == LABEL_BENIGN, "anomaly_score"].to_numpy(dtype=float)
    att = tst.loc[tst["label"] == LABEL_ATTACK, "anomaly_score"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    bins = np.linspace(min(ben.min(), att.min() if att.size else 0), max(ben.max(), att.max() if att.size else 1), 60)
    ax.hist(ben, bins=bins, alpha=0.65, density=True, label=f"benign (n = {len(ben)})", color="#2a6f97")
    if att.size:
        ax.hist(att, bins=bins, alpha=0.6, density=True, label=f"attack (n = {len(att)})", color="#c1121f")
    ax.axvline(float(tst["threshold"].iloc[0]), color="#000", ls="--", lw=1.2,
               label=f"calibrated threshold = {tst['threshold'].iloc[0]:.4f}")
    ax.set_xlabel("anomaly score  s = 1 - max cosine similarity to normal memory  (dimensionless)")
    ax.set_ylabel("probability density")
    ax.set_title("Figure 4: Normal vs anomalous score distribution")
    ax.legend()
    _save(fig, out_dir, "fig04_score_distribution.png", status)


def figure_confusion(pred: pd.DataFrame, out_dir: str, status: Dict[str, str], unit: str = "alert") -> None:
    from sklearn.metrics import confusion_matrix

    plt = _plt()
    tst = pred[pred["split"] == SPLIT_TEST]
    y = (tst["label"] == LABEL_ATTACK).astype(int).to_numpy()
    col = "confirmed_alert" if unit == "alert" else "is_anomaly"
    cm = confusion_matrix(y, tst[col].astype(int).to_numpy(), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["benign", "attack"])
    ax.set_yticks([0, 1], ["benign", "attack"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#10233a", fontsize=11)
    ax.set_title(f"Figure 5: Confusion matrix ({unit}-level, n = {len(tst)})")
    fig.colorbar(im, ax=ax, fraction=0.046, label="windows")
    _save(fig, out_dir, "fig05_confusion_matrix.png", status)


def figure_latency(bench: Dict[str, Any], out_dir: str, status: Dict[str, str],
                   target_ms: Optional[float] = None) -> None:
    """Figure 6 - distribution of measured per-sample Member-3 latency (ms)."""
    plt = _plt()
    raw = bench.get("raw_ms", {})
    if not raw:
        status["fig06_latency_distribution.png"] = f"{NOT_RUN}: no per-sample timings captured"
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for label, key, col in (("FAISS retrieval", "B_faiss_retrieval", "#2a6f97"),
                            ("threshold + confidence", "C_thresholding", "#e9c46a"),
                            ("Member-3 total", "E_member3_total", "#c1121f")):
        v = np.asarray(raw.get(key, []), dtype=float)
        if v.size == 0:
            continue
        ax.hist(v, bins=30, alpha=0.55, density=True, histtype="step", linewidth=1.4, color=col,
                label=f"{label} (median {np.median(v):.3f} ms, n = {v.size})")
    if target_ms is not None:
        ax.axvline(float(target_ms), ls="--", c="#444", lw=1.1,
                   label=f"design target {float(target_ms):.1f} ms - stated target, not a measured result")
    ax.set_xlabel("per-sample latency (milliseconds)")
    ax.set_ylabel("probability density")
    ax.set_title("Figure 6: Member-3 inference latency distribution")
    ax.legend(loc="upper right", fontsize=7.0)
    _save(fig, out_dir, "fig06_latency_distribution.png", status)


def figure_bar_ablation(df: pd.DataFrame, x: str, metrics: Sequence[str], title: str, fname: str,
                        out_dir: str, status: Dict[str, str], ylab: str = "metric value",
                        ylim01: bool = True, note_col: Optional[str] = None) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(4.4, 1.5 * len(df) + 2.2), 3.4))
    xs = np.arange(len(df))
    w = 0.8 / max(len(metrics), 1)
    for i, m in enumerate(metrics):
        if m not in df.columns:
            continue
        v = pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=float)
        ax.bar(xs + i * w - 0.4 + w / 2, v, width=w, label=m.replace("_", " "))
    ax.set_xticks(xs, [str(t) for t in df[x]])
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(ylab)
    if ylim01:
        ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(ncol=min(len(metrics), 4), fontsize=7.6)
    if note_col and note_col in df.columns:
        ax.text(0.0, -0.22, "; ".join(f"{a}={b}" for a, b in zip(df[x], df[note_col])), transform=ax.transAxes,
                fontsize=6.6, color="#444")
    _save(fig, out_dir, fname, status)


def figure_fidelity(fid_df: pd.DataFrame, summary: Dict[str, Any], out_dir: str, status: Dict[str, str]) -> None:
    """Figure 10 - masking-and-recompute evidence that the attribution is causal."""
    plt = _plt()
    if len(fid_df) == 0:
        status["fig10_fidelity_plus.png"] = f"{NOT_RUN}: no Fidelity+ rows"
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    meths = [m for m in ("attention", "random", "lowest_attention") if m in set(fid_df["masking_method"])]
    xs = np.arange(len(meths))
    before = [float(fid_df[fid_df["masking_method"] == m]["anomaly_score_before"].mean()) for m in meths]
    after = [float(fid_df[fid_df["masking_method"] == m]["anomaly_score_after"].mean()) for m in meths]
    w = 0.36
    axes[0].bar(xs - w / 2, before, width=w, label="before masking", color="#2a6f97")
    axes[0].bar(xs + w / 2, after, width=w, label="after masking", color="#c1121f")
    axes[0].set_xticks(xs, [m.replace("_", "\n") for m in meths])
    axes[0].set_ylabel("mean anomaly score (dimensionless)")
    axes[0].set_xlabel("masking basis (top-K components removed)")
    axes[0].set_title("10a: anomaly score before vs after masking", fontsize=9.5)
    axes[0].legend(fontsize=7.4, loc="upper center", framealpha=0.95)
    vals = np.array([float(summary.get(m, {}).get("mean", np.nan)) for m in meths])
    lo = np.array([float(summary.get(m, {}).get("ci95_low", np.nan)) for m in meths])
    hi = np.array([float(summary.get(m, {}).get("ci95_high", np.nan)) for m in meths])
    err = np.vstack([np.abs(vals - lo), np.abs(hi - vals)])
    axes[1].bar(xs, vals, yerr=err, capsize=4, color=["#2a6f97", "#9aa0a6", "#e9c46a"][: len(meths)])
    axes[1].axhline(0, c="#444", lw=0.8)
    axes[1].set_xticks(xs, [m.replace("_", "\n") for m in meths])
    axes[1].set_ylabel("Fidelity+ (% anomaly-score reduction)")
    axes[1].set_xlabel("masking basis (top-K components removed)")
    axes[1].set_title(f"10b: Fidelity+ with bootstrap 95% CI (n = {fid_df['sample_id'].nunique()} alerts)",
                      fontsize=9.5)
    fig.suptitle("Figure 10: Fidelity+ validation of the graph attribution", fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_dir, "fig10_fidelity_plus.png", status)


def figure_family_breakdown(fam_df: pd.DataFrame, out_dir: str, status: Dict[str, str]) -> None:
    plt = _plt()
    if fam_df is None or len(fam_df) == 0:
        status["fig12_attack_family_performance.png"] = f"{NOT_RUN}: no attack-family rows (labels/attacks missing)"
        return
    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(fam_df) + 2.0))
    y = np.arange(len(fam_df))
    for i, (m, col) in enumerate((("f1", "#2a6f97"), ("precision", "#e9c46a"), ("recall", "#c1121f"))):
        v = pd.to_numeric(fam_df[m], errors="coerce").to_numpy(dtype=float)
        ax.barh(y + (i - 1) * 0.26, v, height=0.24, label=m, color=col)
    ax.set_yticks(y, [f"{r.attack_family} (n={int(r.n_positive)})" for r in fam_df.itertuples()])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("score (unseen family held out of every permitted stage)")
    ax.set_title("Figure 12: Per-attack-family zero-day performance")
    ax.legend(fontsize=7.6, loc="lower right")
    _save(fig, out_dir, "fig12_attack_family_performance.png", status)


def figure_cross_dataset(df: pd.DataFrame, out_dir: str, status: Dict[str, str]) -> None:
    if df is None or len(df) <= 1:
        status["fig11_cross_dataset_comparison.png"] = (
            f"{NOT_RUN}: needs >= 2 test datasets (found {0 if df is None else len(df)}) - "
            "add NF-ToN-IoT-v3 records to run this")
        return
    figure_bar_ablation(df, x="dataset", metrics=("f1", "auprc", "roc_auc", "precision"),
                        title="Figure 11: Cross-dataset generalization", fname="fig11_cross_dataset_comparison.png",
                        out_dir=out_dir, status=status)


def write_metrics(results: Dict[str, Any], out_dir: str) -> None:
    """metrics.json + metrics.csv.  Internal run state (keys starting with ``_``) is excluded:
    published metric files contain metrics, and provenance, and nothing else."""
    clean = {k: v for k, v in dict(results).items() if not str(k).startswith("_")}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(clean, fh, indent=2, default=str)
    flat = flatten(clean)
    pd.DataFrame(flat).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)


def flatten(d: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            rows.extend(flatten(v, key))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            rows.append({"metric": key, "value": f"<{len(v)} rows>"})
        else:
            rows.append({"metric": key, "value": v})
    return rows


def write_manifest(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str, sort_keys=False)
