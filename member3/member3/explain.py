"""Explainable graph attribution (GATv2 structural attention) + Fidelity+ validation.

Attribution is *orchestrated* here, never computed from scratch: the node/edge
attention tensors come from the Member-2 adapter
(``encode_with_attention``), and Member 3 maps them back to host/IP identifiers,
flow records and feature names, then validates the explanation causally by
mask-and-recompute (Fidelity+).

Fidelity+ definition used everywhere in this project (§14):

    Fidelity+ (%) = 100 * (s_before - s_after) / max(s_before, eps)

where s_* is the *same* anomaly score as in inference.py (1 - max cosine
similarity to the normal memory).  Positive values mean removing the
explanation-selected components lowers the anomaly score, i.e. those components
were genuinely driving the alert.  Negative values are reported, not clipped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contract import EPS_DEFAULT, mask_graph_sequence


def validate_attention_payload(payload: Any, graph_sequence: Dict[str, Any], dim: int,
                               who: str = "encode_with_attention") -> Dict[str, Any]:
    """Check the Member-2 attention payload against the graph it came from (contract §C).

    Cheap and worth it: a length mismatch between ``node_attention`` and the graph's nodes is an
    off-by-something that would otherwise surface as a cryptic IndexError deep inside attribution
    (or, worse, silently mislabel which host caused an alert).
    """
    from .contract import ContractError

    if not isinstance(payload, dict):
        raise ContractError(f"adapter.{who}() must return a dict with keys "
                            "{'embedding','node_attention','edge_attention','graph_metadata'}")
    for key in ("embedding", "node_attention", "graph_metadata"):
        if key not in payload:
            raise ContractError(f"adapter.{who}() payload is missing '{key}' (contract §C)")
    z = np.asarray(payload["embedding"], dtype=np.float64)
    if z.ndim > 1:
        z = z.reshape(-1)
    if int(dim) > 0 and z.size != int(dim):
        raise ContractError(f"adapter.{who}(): embedding has {z.size} dims, adapter metadata declares {dim}")
    snaps = (graph_sequence or {}).get("snapshots") or []
    n_nodes = max([len(np.asarray(s.get("x", []))) for s in snaps], default=0)
    n_edges = max([int(np.asarray(s.get("edge_index", np.zeros((2, 0)))).shape[-1]) for s in snaps], default=0)
    na = np.asarray(payload["node_attention"], dtype=np.float64).reshape(-1)
    if na.size == 0 or na.size > max(n_nodes, 1):
        raise ContractError(
            f"adapter.{who}(): 'node_attention' has {na.size} entries but the widest snapshot has "
            f"{n_nodes} nodes - attention must be one weight per node, in node-index order "
            "(contract §C: 'node_attention [N_t]')")
    if not np.all(np.isfinite(na)):
        raise ContractError(f"adapter.{who}(): 'node_attention' contains non-finite values")
    if na.max() > 1.0 + 1e-6 and abs(float(na.sum()) - 1.0) > 1e-3:
        raise ContractError(
            f"adapter.{who}(): 'node_attention' is neither softmax-normalised (sums to {na.sum():.3f}) nor in "
            "[0,1]; Member 3 needs comparable shares to rank and mask nodes")
    if "edge_attention" in payload and payload["edge_attention"] is not None:
        ea = np.asarray(payload["edge_attention"], dtype=np.float64).reshape(-1)
        if ea.size > max(n_edges, 1):
            raise ContractError(f"adapter.{who}(): 'edge_attention' has {ea.size} entries but the widest "
                               f"snapshot has {n_edges} edges")
    gmeta = payload["graph_metadata"] or {}
    for key, width in (("node_ids", n_nodes), ("edge_record_ids", n_edges)):
        v = gmeta.get(key)
        if v is None:
            continue
        flat = v[0] if (isinstance(v, list) and v and isinstance(v[0], (list, tuple))) else v
        if len(flat) > max(width, 1):
            raise ContractError(f"adapter.{who}(): graph_metadata['{key}'] has {len(flat)} ids but the graph "
                                f"has at most {width} {'nodes' if key == 'node_ids' else 'edges'} - ids and "
                                "indices are misaligned (contract §C)")
    return payload


def _as_1d(a: Any) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64)
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim > 1:  # (T, n) or (heads, n) -> aggregate
        x = x.mean(axis=tuple(range(x.ndim - 1)))
    return x.ravel()


def _rank(idx_weights: Sequence[Tuple[Any, float]], top_k: int) -> List[Tuple[Any, float]]:
    return sorted(idx_weights, key=lambda t: (-t[1], str(t[0])))[: max(int(top_k), 0)]


def aggregate_attention(expl: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    node_attn = _as_1d(expl.get("node_attention", []))
    edge_attn = _as_1d(expl.get("edge_attention", []))
    if node_attn.size == 0:
        raise RuntimeError(
            "attention extraction unavailable: adapter returned empty node_attention. "
            "Member 2 must expose per-node structural attention (handoff contract §C)."
        )
    return node_attn, edge_attn, dict(expl.get("graph_metadata") or {})


def build_explanation(adapter: Any, graph_sequence: Dict[str, Any], score: float, threshold: float,
                      confidence: float, alert_id: str, k_nodes: int = 5, k_edges: int = 5,
                      k_features: int = 8, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compact, human-readable explanation object for one alert (§13)."""
    if graph_sequence is None:
        raise ValueError("build_explanation needs the original graph sequence (contract §E); "
                         "cached embeddings alone cannot be explained")
    out = validate_attention_payload(adapter.encode_with_attention(graph_sequence), graph_sequence,
                                     int(getattr(adapter, "embedding_dim", 0)
                                         or (adapter.metadata() or {}).get("embedding_dim", 0)))
    node_attn, edge_attn, gmeta = aggregate_attention(out)
    node_ids = gmeta.get("node_ids") or []
    edge_ids = gmeta.get("edge_record_ids") or []
    feature_names = gmeta.get("feature_names") or []

    n_snaps = len(graph_sequence.get("snapshots", []))
    node_keys = [f"snapshot_node_{i}" for i in range(len(node_attn))]
    if node_ids:
        flat = node_ids[0] if isinstance(node_ids[0], (list, tuple)) else node_ids
        node_keys = [str(v) for v in list(flat)[: len(node_attn)]] + [f"node_{i}" for i in range(len(node_keys) - len(flat), len(node_attn))]
    top_nodes = _rank([(i, float(node_attn[i])) for i in range(len(node_attn))], k_nodes)
    top_edges = _rank([(i, float(edge_attn[i])) for i in range(len(edge_attn))], k_edges)

    def _label(i: int, w: float) -> Dict[str, Any]:
        return {"node_index": int(i), "host": node_keys[i] if i < len(node_keys) else f"node_{i}",
                "attention": round(w, 6), "attention_share": round(float(w / (node_attn.sum() + 1e-12)), 6)}

    def _edge(i: int, w: float) -> Dict[str, Any]:
        rec = None
        if isinstance(edge_ids, (list, tuple)) and len(edge_ids):
            src = edge_ids[0] if isinstance(edge_ids[0], (list, tuple)) else edge_ids
            rec = src[i] if i < len(src) else None
        return {"edge_index": int(i), "flow_record": rec, "attention": round(float(w), 6)}

    # feature attribution only where upstream information permits it (§13.5)
    feat_src, top_features = "unavailable", []
    feats = np.asarray([])
    if "snapshots" in graph_sequence and graph_sequence["snapshots"]:
        ea = np.asarray(graph_sequence["snapshots"][0].get("edge_attr", []), dtype=np.float64)
        if ea.size:
            w = edge_attn if edge_attn.shape[0] == ea.shape[0] else np.ones(ea.shape[0])
            feats = (np.abs(ea) * (w / (w.sum() + 1e-12))[:, None]).sum(axis=0)
            feat_src = "derived:attention_weighted_edge_feature_magnitude"
    if feats.size:
        names = [str(feature_names[i]) if i < len(feature_names) else f"feature_{i}" for i in range(len(feats))]
        top_features = [{"feature": names[i], "attribution_weight": round(float(feats[i]), 6)}
                        for i, _ in _rank([(i, float(feats[i])) for i in range(len(feats))], k_features)]

    coverage = float(np.sum([w for _, w in top_nodes]) / (node_attn.sum() + 1e-12)) if len(node_attn) else float("nan")
    expl = {
        "alert_id": str(alert_id),
        "anomaly_score": round(float(score), 6),
        "threshold": round(float(threshold), 6),
        "confidence": round(float(confidence), 4),
        "exceeds_threshold": bool(score > threshold),
        "top_nodes": [_label(i, w) for i, w in top_nodes],
        "top_edges": [_edge(i, w) for i, w in top_edges],
        "top_features": top_features,
        "feature_attribution_source": feat_src,
        "temporal_snapshot": int(gmeta.get("primary_snapshot", gmeta.get("snapshot_ids", [0])[0] if gmeta.get("snapshot_ids") else 0)),
        "n_snapshots": int(n_snaps),
        "explanation_coverage_topk_nodes": round(coverage, 4),
        "nearest_normal_similarity": (extra or {}).get("nearest_normal_similarity"),
        "memory_source_dataset": (extra or {}).get("memory_source_dataset"),
        "summary": None,
    }
    expl["summary"] = _summarize(expl)
    return expl


def _summarize(expl: Dict[str, Any]) -> str:
    if expl["top_nodes"]:
        hosts = ", ".join(str(t["host"]) for t in expl["top_nodes"][:3])
        return (f"anomaly score {expl['anomaly_score']:.3f} > threshold {expl['threshold']:.3f}; "
                f"most influential hosts: {hosts}")
    return f"anomaly score {expl['anomaly_score']:.3f}; no attention information available"


# --------------------------------------------------------------------------- #
# Fidelity+ masking experiment
# --------------------------------------------------------------------------- #
def _mask_sequence(adapter: Any, seq: Dict[str, Any], nodes: Sequence[int], edges: Sequence[int],
                   kind: str) -> Dict[str, Any]:
    if kind == "nodes":
        if hasattr(adapter, "mask_nodes"):
            return adapter.mask_nodes(seq, nodes)
        return mask_graph_sequence(seq, node_ids=nodes)
    if hasattr(adapter, "mask_edges"):
        return adapter.mask_edges(seq, edges)
    return mask_graph_sequence(seq, edge_ids=edges)


def _select_components(attn: np.ndarray, k: int, mode: str, rng: np.random.Generator) -> List[int]:
    n = len(attn)
    k = max(1, min(int(k), n))
    if mode == "random":
        return sorted(int(i) for i in rng.choice(n, size=k, replace=False))
    order = np.argsort(-attn, kind="stable")
    if mode == "attention":
        return sorted(int(i) for i in order[:k])
    if mode == "lowest_attention":
        return sorted(int(i) for i in order[-k:])
    raise ValueError(f"unknown masking selection '{mode}'")


def fidelity_plus(scorer: Any, samples: pd.DataFrame, graphs: Dict[str, Dict[str, Any]], cfg: Dict[str, Any],
                  methods: Sequence[str] = ("attention", "random", "lowest_attention")) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Mask-and-recompute validation of the attribution (§14).

    For every evaluated alert, the top-K explanation components are removed from the
    temporal graph, the embedding is recomputed through the Member-2 adapter and the
    anomaly score is re-evaluated against the *frozen* memory/threshold.
    """
    xc = dict(cfg.get("xai", {}))
    k = int(xc.get("mask_top_k", 3))
    kind = str(xc.get("mask_target", "nodes"))
    if kind not in ("nodes", "edges"):
        raise ValueError(f"xai.mask_target must be 'nodes' or 'edges', got '{kind}'")
    adapter = scorer.adapter
    if not hasattr(adapter, "encode_with_attention"):
        raise RuntimeError("Fidelity+ blocked: adapter has no encode_with_attention() (contract §C)")
    mem_frozen = scorer.memory.stats.immutable
    scorer.memory.set_immutable(True)
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    skipped_no_graph = 0
    try:
        for _, r in samples.iterrows():
            sid = str(r["sample_id"])
            seq = graphs.get(sid)
            if seq is None:
                skipped_no_graph += 1
                continue
            base = scorer.score_embeddings(np.asarray([r["embedding"]], dtype=np.float64), with_neighbours=False)
            s0 = float(base["anomaly_score"][0])
            expl = validate_attention_payload(adapter.encode_with_attention(seq), seq,
                                              scorer.adapter_meta.get("embedding_dim", 0),
                                              who="encode_with_attention")
            node_attn, edge_attn, _ = aggregate_attention(expl)
            attn = node_attn if kind == "nodes" else (edge_attn if edge_attn.size else node_attn)
            for m in methods:
                comp = _select_components(attn, k, m, np.random.default_rng(int(rng.integers(0, 1 << 31))))
                masked = _mask_sequence(adapter, seq, comp if kind == "nodes" else [],
                                       comp if kind == "edges" else [], kind)
                z = np.asarray(adapter.encode(masked), dtype=np.float64).reshape(1, -1)
                s1 = float(scorer.score_embeddings(z, with_neighbours=False)["anomaly_score"][0])
                mass = float(np.sum(attn[comp]) / (attn.sum() + 1e-12)) if len(comp) else float("nan")
                rows.append({"sample_id": sid, "alert_id": f"{sid}#{m}", "masking_method": m, "mask_target": kind,
                             "n_components_masked": len(comp), "anomaly_score_before": s0,
                             "anomaly_score_after": s1, "score_reduction": s0 - s1,
                             "fidelity_plus_pct": 100.0 * (s0 - s1) / max(abs(s0), EPS_DEFAULT),
                             "attention_coverage": mass, "dataset_id": str(r.get("dataset_id")),
                             "attack_family": str(r.get("attack_family"))})
    finally:
        scorer.memory.set_immutable(mem_frozen)
    if not rows:
        raise RuntimeError(
            f"Fidelity+ NOT RUN: no graph sequences available for the {len(samples)} evaluated alerts "
            f"(skipped {skipped_no_graph}). Member 2 must hand over sample graphs (contract §E/§H)."
        )
    df = pd.DataFrame(rows)
    summary: Dict[str, Any] = {}
    boot = int(cfg.get("statistics", {}).get("bootstrap_iterations", 1000))
    for m, g in df.groupby("masking_method"):
        v = g["fidelity_plus_pct"].to_numpy(dtype=float)
        ci = bootstrap_ci(v, n_boot=boot, seed=int(cfg.get("seed", 0)))
        summary[str(m)] = {"n_alerts": int(len(g)), "mean": float(v.mean()), "median": float(np.median(v)),
                          "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                          "ci95_low": ci["low"], "ci95_high": ci["high"],
                          "frac_score_drops": float(np.mean(v > 0)),
                          "mean_before": float(g["anomaly_score_before"].mean()),
                          "mean_after": float(g["anomaly_score_after"].mean())}
    summary["_meta"] = {"mask_target": kind, "top_k": k, "n_samples_without_graph": int(skipped_no_graph),
                        "eps": EPS_DEFAULT,
                        "formula": "100*(s_before - s_after)/max(|s_before|,eps), s = 1 - max cosine sim to normal memory"}
    if "attention" in summary and "random" in summary:
        summary["_meta"]["attention_minus_random_mean_pct"] = float(
            summary["attention"]["mean"] - summary["random"]["mean"])
    return df, summary


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0, statistic: str = "mean",
                 alpha: float = 0.05) -> Dict[str, float]:
    """Percentile bootstrap CI, seeded, with re-sampling over observations."""
    v = np.asarray(list(values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"point": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0, "n_boot": 0}
    rng = np.random.default_rng(seed)
    stats: List[float] = []
    for _ in range(int(n_boot)):
        s = rng.choice(v, size=v.size, replace=True)
        stats.append(float(np.mean(s)) if statistic == "mean" else float(np.median(s)))
    arr = np.array(stats)
    return {"point": float(np.mean(v) if statistic == "mean" else np.median(v)),
            "low": float(np.percentile(arr, 100 * alpha / 2)), "high": float(np.percentile(arr, 100 * (1 - alpha / 2))),
            "n": int(v.size), "n_boot": int(n_boot)}
