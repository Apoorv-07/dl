#!/usr/bin/env python3
"""Single-command Member-3 experiment runner.

    python run_experiment.py --config config.yaml
    python run_experiment.py --config config.yaml --stage calibration|evaluation|xai|ablation

Stages are ordered and each stage consumes only artefacts saved by earlier ones, so a
stage can be re-run alone (``calibration`` must exist first).  All outputs land in
``results/`` and ``figures/``; ``results/manifest.json`` is rewritten by every stage and
always records the status and failure reason of each stage (no hidden failures).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from member3 import evaluate as ev  # noqa: E402
from member3.contract import (  # noqa: E402
    SPLIT_CALIBRATION, SPLIT_MEMORY, SPLIT_TEST, ContractError, assert_splits_clean, git_commit,
    load_adapter, load_graphs, load_records, set_seed, sha256_file, software_versions, validate_records,
)
from member3.explain import build_explanation, fidelity_plus  # noqa: E402
from member3.inference import Member3Scorer, benchmark_latency, hardware_info  # noqa: E402

STAGES = ("calibration", "evaluation", "xai", "ablation")


# --------------------------------------------------------------------------- #
# runner context
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, cfg: Dict[str, Any], cfg_path: str):
        self.cfg, self.cfg_path = cfg, cfg_path
        self.d = cfg.get("output", {})
        self.results_dir = str(self.d.get("results_dir", "results"))
        self.figures_dir = str(self.d.get("figures_dir", "figures"))
        self.state_dir = str(self.d.get("state_dir", "state"))
        self.config_id = str(cfg.get("experiment", {}).get("id", "member3-default"))
        self.experiment_name = str(cfg.get("experiment", {}).get("name", "member3_full_run"))
        for p in (self.results_dir, self.figures_dir, self.state_dir):
            os.makedirs(p, exist_ok=True)
        self.fig_status: Dict[str, str] = {}
        self.results: Dict[str, Any] = {"_tidy_part": []}
        self.stage_log: Dict[str, Dict[str, Any]] = {}
        self._adapter: Optional[Any] = None
        self._adapter_meta: Optional[Dict[str, Any]] = None
        self._records: Optional[pd.DataFrame] = None
        self._graphs: Optional[Dict[str, Dict[str, Any]]] = None
        self._scorer: Optional[Member3Scorer] = None

    # ---- paths ----------------------------------------------------------- #
    def path(self, kind: str, name: str) -> str:
        base = {"results": self.results_dir, "figures": self.figures_dir, "state": self.state_dir}[kind]
        return os.path.join(base, name)

    # ---- data / model --------------------------------------------------- #
    @property
    def data_source(self) -> str:
        """Which dataset the reported numbers describe - written into every table row."""
        e = dict(self.cfg.get("evaluation", {}))
        return str(self.cfg.get("data", {}).get("dataset_label")
                   or ",".join([str(x) for x in (e.get("test_datasets") or [])]) or "undeclared")

    def records(self) -> pd.DataFrame:
        """Member-1/2 records export.  Member 3 never generates data of its own."""
        if self._records is None:
            d = dict(self.cfg.get("data", {}))
            rp = d.get("records_path")
            if not rp or not os.path.exists(str(rp)):
                raise ContractError(
                    "data.records_path is not set or does not exist.\n"
                    "Member 3 has no fallback data: point config.yaml at the records export described in "
                    "docs/MEMBER2_HANDOFF_CONTRACT.md (Member 1 supplies graphs + splits + stream ids, "
                    "Member 2 supplies the embeddings).")
            df = load_records(str(rp))
            validate_records(df, expected_dim=int(self.cfg.get("member2", {}).get("embedding_dim", 0) or 0),
                             need_labels=bool(d.get("require_labels", True)))
            derived_streams = "stream_id" not in df.columns
            if derived_streams:
                from member3.contract import assign_streams
                df = assign_streams(df, window_seconds=float(d.get("window_seconds", 30)),
                                    seed=int(self.cfg.get("seed", 0)))
            ev_cfg = dict(self.cfg.get("evaluation", {}))
            permitted = [str(ev_cfg["train_dataset"])] if ev_cfg.get("train_dataset") else None
            assert_splits_clean(df, permitted_reference_datasets=permitted)
            self.results["_provenance"] = {
                "record_file": str(rp),
                "record_file_sha256": sha256_file(str(rp))[:16],
                "graph_file": str(d.get("graph_path") or "not provided (Fidelity+ will be NOT RUN)"),
                "stream_order_source": "member1/2 columns" if not derived_streams
                                       else "derived by member3.contract.assign_streams (upstream supplied none)",
                "permitted_reference_datasets": permitted or "not declared in config (dataset audit skipped)",
                "rows": {k: int(v) for k, v in df["split"].value_counts().items()}}
            self._records = df
        return self._records

    def graphs(self) -> Dict[str, Dict[str, Any]]:
        if self._graphs is None:
            gp = dict(self.cfg.get("data", {})).get("graph_path")
            if gp and os.path.exists(str(gp)):
                self._graphs = load_graphs(str(gp))
            else:
                self._graphs = {}
        return self._graphs

    def adapter(self) -> Tuple[Any, Dict[str, Any]]:
        if self._adapter is None:
            a, meta = load_adapter(self.cfg)
            self._adapter, self._adapter_meta = a, meta
            self.results["_adapter_meta"] = meta
        return self._adapter, self._adapter_meta or {}

    # ---- scorer state --------------------------------------------------- #
    def build_scorer(self, cfg: Optional[Dict[str, Any]] = None, persist: bool = False) -> Member3Scorer:
        cfg = cfg or self.cfg
        adapter, meta = self.adapter()
        scorer = Member3Scorer.build(cfg, adapter, meta, self.records())
        # only the canonical calibration scorer becomes the run state; ablation arms build their
        # own throw-away banks and must not leak into the manifest (auditability, brief sec.30)
        if persist:
            self._scorer = scorer
            idx = self.path("state", "memory_index.bin")
            scorer.memory.save(idx)
            with open(self.path("state", "threshold.json"), "w") as fh:
                json.dump(scorer.threshold.as_dict(), fh, indent=2)
            with open(self.path("state", "adapter_meta.json"), "w") as fh:
                json.dump(meta, fh, indent=2)
        return scorer

    def load_scorer(self) -> Member3Scorer:
        if self._scorer is not None:
            return self._scorer
        idx = self.path("state", "memory_index.bin")
        if not os.path.exists(idx):
            return self.build_scorer(persist=True)
        from member3.memory import AdaptiveMemoryBank

        adapter, meta = self.adapter()
        bank = AdaptiveMemoryBank.load(idx)
        with open(self.path("state", "threshold.json")) as fh:
            thr = ev_threshold_from_dict(json.load(fh))
        self._scorer = Member3Scorer(adapter, meta, bank, thr, self.cfg,
                                    provenance={"memory_source": "reloaded_from_state",
                                                "calibration_source": thr.calibrated_on})
        return self._scorer

    # ---- io ------------------------------------------------------------- #
    def write_csv(self, name: str, df: pd.DataFrame) -> None:
        df.to_csv(self.path("results", name), index=False)
        print(f"[out] {self.path('results', name)}  ({len(df)} rows)")

    def update_manifest(self, stage: str, ok: bool, secs: float, error: Optional[str] = None) -> None:
        self.stage_log[stage] = {"status": "ok" if ok else "failed", "seconds": round(secs, 2),
                                 "error": error, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.results["_stages"] = self.stage_log
        write_manifest(self, stage)


def ev_threshold_from_dict(d: Dict[str, Any]):
    from member3.inference import ThresholdModel

    keep = {k: v for k, v in d.items() if k in ThresholdModel.__dataclass_fields__}
    return ThresholdModel(**keep)


def write_manifest(run: Runner, stage: str) -> None:
    sc = run._scorer
    thr = sc.threshold.as_dict() if sc else run.results.get("_threshold", {})
    mem = sc.memory.summary() if sc else run.results.get("_memory_summary", {})
    fid = run.results.get("xai", {}).get("fidelity_plus", {})
    ds = run.records() if run._records is not None else None
    manifest = {
        "experiment_name": run.experiment_name,
        "config_id": run.config_id,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        "git_commit": git_commit(),
        "data_source": run.data_source,
        "dataset_identifiers": sorted(ds["dataset_id"].unique().tolist()) if ds is not None else "NOT RUN",
        "split_sizes": {f"{a}/{b}": int(c) for (a, b), c in ds.groupby(["split", "label"]).size().items()} if ds is not None else {},
        "counts": {k: int(v) for k, v in (ds["split"].value_counts().items() if ds is not None else [])},
        "model_checkpoint": {k: run.results.get("_adapter_meta", {}).get(k) for k in
                             ("architecture_id", "checkpoint_version", "framework", "framework_version", "device")},
        "checkpoint_checksum": _checksum(run.cfg.get("member2", {}).get("artifacts_dir", "member2_artifacts")),
        "embedding_dim": run.results.get("_adapter_meta", {}).get("embedding_dim"),
        "threshold_method": thr.get("method"), "threshold_value": thr.get("value"),
        "threshold_scale": thr.get("scale"), "threshold_calibration_source": thr.get("calibrated_on"),
        "memory_size": mem.get("n"), "memory_capacity": mem.get("capacity"),
        "memory_selection": mem.get("selection"), "memory_version": mem.get("version"),
        "faiss": {"index_type": mem.get("index_type"), "hnsw_M": mem.get("hnsw_M"),
                  "ef_construction": mem.get("ef_construction"), "ef_search": mem.get("ef_search"),
                  "backend": mem.get("index_backend")},
        "k": mem.get("k"), "debounce_length": run.cfg.get("alerting", {}).get("debounce_steps"),
        "adaptation_enabled": bool(run.cfg.get("adaptation", {}).get("enabled", True)),
        "n_update_events": mem.get("n_update_events"),
        "scoring_definition": "anomaly_score = 1 - max_j cos(z, m_j) over top-k normal memory entries (L2-normalised z)",
        "neighbor_aggregation": run.cfg.get("retrieval", {}).get("neighbor_aggregation", "max"),
        "masking_target": run.cfg.get("xai", {}).get("mask_target"),
        "fidelity_plus_methods": list(fid.keys()) if fid else "NOT RUN",
        "random_seed": run.cfg.get("seed"),
        "software_versions": software_versions(),
        "hardware": hardware_info(),
        "stages": run.stage_log,
        "reproducibility": {"command": f"python run_experiment.py --config {os.path.basename(run.cfg_path)}",
                            "config_path": run.cfg_path,
                            "config_sha256": sha256_file(run.cfg_path) if os.path.exists(run.cfg_path) else None},
        "provenance": run.results.get("provenance", run.results.get("_provenance", {})),
    }
    manifest["results"] = _prune(run.results)
    ev.write_manifest(run.path("results", "manifest.json"), manifest)


def _checksum(artifacts_dir: str) -> Optional[str]:
    for name in ("checkpoint.pt", "checkpoint.bin", "model.pt"):
        p = os.path.join(artifacts_dir, name)
        if os.path.exists(p):
            return sha256_file(p)[:16]
    return "not-provided"


def _prune(results: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in results.items():
        if isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items() if not isinstance(vv, (pd.DataFrame, np.ndarray))}
        elif isinstance(v, (pd.DataFrame, np.ndarray)):
            out[k] = f"<{type(v).__name__} rows={getattr(v, 'shape', (len(v),))[0]}>"
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_calibration(run: Runner) -> None:
    scorer = run.build_scorer(persist=True)
    run.results["_threshold"] = scorer.threshold.as_dict()
    run.results["_memory_summary"] = scorer.memory.summary()
    print(f"[calibration] memory n={scorer.memory.stats.n}/{scorer.memory.stats.capacity} "
          f"index={scorer.memory.stats.index_type} k={scorer.k} | "
          f"threshold[{scorer.threshold.method}]={scorer.threshold.value:.6f} "
          f"(n_cal={scorer.threshold.n_calibration}, scale={scorer.threshold.scale:.6f})")
    run.results["provenance"] = {"memory_split": SPLIT_MEMORY, "calibration_split": SPLIT_CALIBRATION,
                                 "test_split_used_for_fitting": False, **scorer.provenance,
                                 **run.results.get("_provenance", {})}


def stage_evaluation(run: Runner) -> None:
    scorer = run.load_scorer()
    scorer.memory.set_immutable(True)
    ws = float(run.cfg.get("data", {}).get("window_seconds", 30))
    records = run.records()
    pred = scorer.predict(records)
    pred = scorer.verify_alerts(pred, window_seconds=ws)
    pred.attrs["window_seconds"] = ws
    test = pred[pred["split"] == SPLIT_TEST]
    if len(test) == 0:
        raise ContractError("evaluation aborted: no test-split samples found")

    overall = ev.overall_metrics(pred, run.cfg, unit="sample")
    overall_alert = ev.overall_metrics(pred, run.cfg, unit="alert")
    per_dataset: List[Dict[str, Any]] = []
    for ds_id in sorted(test["dataset_id"].unique()):
        sub = pred[pred["dataset_id"] == ds_id]
        m = ev.overall_metrics(sub, run.cfg, dataset_label=str(ds_id), unit="sample")
        m["unit"] = "sample"
        ma = ev.overall_metrics(sub, run.cfg, dataset_label=str(ds_id), unit="alert")
        m["alert_f1"], m["alert_fpr"], m["alert_precision"], m["n_alerts"] = (
            ma["f1"], ma["fpr"], ma["precision"], ma["n_alerts"])
        per_dataset.append(m)
    fam = ev.per_attack_family(pred, run.cfg)
    graphs = run.graphs()
    seq_list = [graphs.get(str(sid)) for sid in test["sample_id"]] if graphs else None
    if seq_list is not None and all(v is None for v in seq_list):
        seq_list = None          # no per-window graphs on disk: skip the encode-stage benchmark
    bench = benchmark_latency(scorer, np.vstack(list(test["embedding"])), graphs=seq_list,
                              n_samples=int(run.cfg.get("benchmark", {}).get("n_samples", 200)),
                              n_repeat=int(run.cfg.get("benchmark", {}).get("repeats", 3)))
    # ---- leakage-audited leave-one-family-out --------------------------- #
    families = sorted(f for f in test.loc[test["label"] == "attack", "attack_family"].unique())
    lfo, audit = run_leave_one_family_out(run, scorer, families)

    run.results["evaluation"] = {"overall_sample_level": overall, "overall_alert_level": overall_alert,
                                "per_dataset": per_dataset, "latency": bench, "lfo_audit": audit,
                                "memory_summary": scorer.memory.summary(), "n_test": int(len(test))}
    run.write_csv("predictions.csv", _prediction_columns(pred))
    run.write_csv("attack_family_results.csv", fam)
    run.write_csv("latency.csv", ev.latency_table(bench, run.config_id, run.data_source))
    run.results["_predictions_index"] = str(list(pred.columns))
    _evaluation_tables_and_figures(run, pred, overall, overall_alert, per_dataset, fam, bench, lfo)


def run_leave_one_family_out(run: Runner, scorer: Member3Scorer, families: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    records = run.records()
    base_vec_hash = _vec_hash(scorer.memory.vectors())
    base_tau = float(scorer.threshold.value)
    n_mem = int((records["split"] == SPLIT_MEMORY).sum())
    rebuild = bool(run.cfg.get("zero_day", {}).get("rebuild_permitted_stages", True)) and n_mem <= 20000
    reason = ("memory+threshold rebuilt per fold with the held-out family removed"
              if rebuild else
              "permitted splits are benign-only, so removing an attack family provably cannot alter them; "
              "baseline memory/threshold reused (zero_day.rebuild_permitted_stages auto-disabled above 20k rows)")

    ws = float(run.cfg.get("data", {}).get("window_seconds", 30))

    def per_family(fam: str) -> Dict[str, Any]:
        """Fold: permitted stages rebuilt without family F; F still evaluated as unseen."""
        if rebuild:
            sc = run.build_scorer(cfg=_with_excluded_family(run.cfg, fam))
            changed = int(_vec_hash(sc.memory.vectors()) != base_vec_hash)
        else:
            sc, changed = scorer, 0
        rec = run.records()
        allowed_reference = rec[rec["attack_family"] != fam]        # family gone from memory+calibration inputs
        if rebuild and allowed_reference is not rec:
            sc = Member3Scorer.build(_with_excluded_family(run.cfg, fam), sc.adapter, sc.adapter_meta, allowed_reference)
        tau = float(sc.threshold.value)
        test_rows = rec[rec["split"] == SPLIT_TEST]
        p = sc.verify_alerts(sc.predict(test_rows), window_seconds=ws)
        ben = p[p["label"] == "benign"]
        pos = p[p["attack_family"] == fam]
        sub = pd.concat([pos, ben]).sort_index()
        y = (sub["attack_family"] == fam).astype(int).to_numpy()
        m = ev.classification_metrics(y, sub["is_anomaly"].astype(int).to_numpy(),
                                      sub["anomaly_score"].to_numpy(dtype=float))
        m.update(ev.score_summary(sub, (sub["attack_family"] == fam).to_numpy()))
        dd = pos["detection_delay_steps"].dropna().to_numpy(dtype=float)
        col = "alert_raised" if "alert_raised" in pos.columns else "confirmed_alert"
        return {"threshold": tau, "memory_changed": changed, "n_samples": int(len(sub)),
                "n_positive": int(len(pos)), "n_benign_negative_windows": int(len(ben)),
                "f1": m["f1"], "precision": m["precision"], "recall": m["recall"], "fpr": m["fpr"],
                "fnr": m["fnr"], "auprc": m["auprc"], "roc_auc": m["roc_auc"],
                "family_alert_rate": float(pos[col].mean()) if len(pos) else float("nan"),
                "mean_detection_delay_steps": float(np.mean(dd)) if dd.size else float("nan"),
                "mean_detection_delay_seconds": float(np.mean(dd) * ws) if dd.size else float("nan"),
                "dataset": ",".join(sorted(p["dataset_id"].unique())), "split": SPLIT_TEST,
                "held_out_family": fam, "evaluation_contrast": "held-out family vs benign windows"}

    lfo, audit = ev.leave_one_family_out(records, dict(run.cfg, _baseline_threshold=base_tau), families, per_family)
    audit["note"] = reason
    audit["held_out_family_explicit"] = True
    if len(lfo):
        lfo["rebuild_permitted_stages"] = rebuild
    return lfo, audit


def _with_excluded_family(cfg: Dict[str, Any], fam: str) -> Dict[str, Any]:
    c = json.loads(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, default=str))
    c.setdefault("zero_day", {})["exclude_attack_family"] = fam
    return c


def _vec_hash(X: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(np.asarray(X, dtype=np.float32)).tobytes()).hexdigest()[:16]


def _prediction_columns(pred: pd.DataFrame) -> pd.DataFrame:
    keep = ["dataset_id", "split", "sample_id", "window_id", "timestamp", "stream_id", "stream_pos", "episode_id",
            "label", "attack_family", "anomaly_score", "nearest_normal_similarity", "threshold", "threshold_method",
            "confidence", "is_anomaly", "consecutive_anomalies", "alert_state", "confirmed_alert", "episode_detected",
            "detection_delay_steps", "detection_delay_seconds", "memory_version", "checkpoint_version",
            "neighbor_sample_ids", "neighbor_similarities"]
    cols = [c for c in keep if c in pred.columns]
    out = pred[cols].copy()
    out["embedding_dim"] = pred["embedding_dim"] if "embedding_dim" in pred else np.nan
    return out


def _evaluation_tables_and_figures(run: Runner, pred: pd.DataFrame, overall: Dict[str, Any],
                                   overall_alert: Dict[str, Any], per_dataset: List[Dict[str, Any]],
                                   fam: pd.DataFrame, bench: Dict[str, Any], lfo: pd.DataFrame) -> None:
    ctx = {"data_source": run.data_source, "config_id": run.config_id, "split": SPLIT_TEST}
    t1 = pd.DataFrame(per_dataset) if per_dataset else pd.DataFrame()
    t0 = pd.DataFrame([{**overall, "dataset": "POOLED_TEST", "split": SPLIT_TEST},
                       {**overall_alert, "dataset": "POOLED_TEST", "split": SPLIT_TEST}])
    bundles = [
        ("TABLE1_overall_cross_dataset", "Overall cross-dataset performance (benign-only reference, unseen test datasets)", t1, ctx),
        ("TABLE2_per_attack_family", "Per-attack-family performance on the test dataset", fam, ctx),
        ("TABLE3_leave_one_family_out", "Leave-one-attack-family-out zero-day results", lfo, ctx),
        ("TABLE4_operational_latency", "Operational latency results (per sample, ms)",
         ev.latency_table(bench, run.config_id, run.data_source), ctx),
    ]
    tidy_df, md = ev.build_paper_tables(bundles)
    run.results["_tidy_part"] = [tidy_df]
    run._tables_md_part1 = md
    ev.figure_architecture(run.figures_dir, run.fig_status)
    ev.figure_roc_pr(pred, run.figures_dir, run.fig_status)
    ev.figure_score_distribution(pred, run.figures_dir, run.fig_status)
    ev.figure_confusion(pred, run.figures_dir, run.fig_status, unit="alert")
    ev.figure_latency(bench, run.figures_dir, run.fig_status,
                      target_ms=float(run.cfg.get("benchmark", {}).get("latency_target_ms", 5.0)))
    run.write_csv("metrics.csv", pd.DataFrame(ev.flatten({"evaluation": run.results.get("evaluation", {})})))



def stage_xai(run: Runner) -> None:
    scorer = run.load_scorer()
    test = scorer.verify_alerts(scorer.predict(run.records()),
                               window_seconds=float(run.cfg.get("data", {}).get("window_seconds", 30)))
    alerts = test[test["confirmed_alert"] & (test["label"] == "attack")]
    if len(alerts) == 0:
        alerts = test[test["is_anomaly"] & (test["label"] == "attack")]
        note = "no confirmed alerts; using single-window anomalies for XAI coverage"
    else:
        note = "confirmed alerts only"
    max_alerts = int(run.cfg.get("xai", {}).get("max_alerts", 60))
    alerts = alerts.sort_values("anomaly_score", ascending=False).head(max_alerts)
    graphs = run.graphs()
    out: Dict[str, Any] = {"note": note, "n_alert_candidates": int(len(alerts))}
    explanations: List[Dict[str, Any]] = []
    if len(alerts):
        for _, r in alerts.iterrows():
            sid = str(r["sample_id"])
            seq = graphs.get(sid)
            if seq is None:
                continue
            expl = build_explanation(scorer.adapter, seq, float(r["anomaly_score"]), float(r["threshold"]),
                                     float(r["confidence"]), alert_id=f"{r['dataset_id']}:{sid}",
                                     k_nodes=int(run.cfg.get("xai", {}).get("top_nodes", 5)),
                                     k_edges=int(run.cfg.get("xai", {}).get("top_edges", 5)),
                                     k_features=int(run.cfg.get("xai", {}).get("top_features", 8)),
                                     extra={"nearest_normal_similarity": float(r["nearest_normal_similarity"]),
                                            "memory_source_dataset": scorer.provenance.get("memory_source")})
            explanations.append(expl)
        with open(run.path("results", "explanations.json"), "w") as fh:
            json.dump(explanations, fh, indent=2, default=str)
        out["explanations_written"] = len(explanations)
    out["attribution_stats"] = ev.explanation_stats(explanations)
    try:
        fid_df, fid_summary = fidelity_plus(scorer, alerts, graphs, run.cfg,
                                            methods=tuple(run.cfg.get("xai", {}).get("methods",
                                                                                      ["attention", "random", "lowest_attention"])))
        run.write_csv("fidelity_plus.csv", fid_df)
        out["fidelity_plus"] = fid_summary
        ev.figure_fidelity(fid_df, fid_summary, run.figures_dir, run.fig_status)
        ctx = {"data_source": run.data_source, "config_id": run.config_id}
        meta = fid_summary.pop("_meta", {})
        rows = [dict(v, masking_method=k, row_level="method", dataset="ALL_TEST_ALERTS",
                     mask_target=meta.get("mask_target"), top_k=meta.get("top_k"),
                     formula=meta.get("formula"),
                     attention_minus_random_mean_pct=meta.get("attention_minus_random_mean_pct"),
                     n_samples_without_graph=meta.get("n_samples_without_graph"))
                for k, v in fid_summary.items() if isinstance(v, dict)]
        fid_summary["_meta"] = meta
        per_fam = (fid_df[fid_df["masking_method"] == fid_df["masking_method"].iloc[0]]
                   .groupby(["attack_family", "masking_method"])["fidelity_plus_pct"]
                   .agg(n_alerts="count", mean="mean", median="median", std=lambda x: x.std(ddof=1)).reset_index())
        per_fam["row_level"] = "method_x_attack_family"
        per_fam["dataset"] = ",".join(sorted(set(fid_df["dataset_id"])))
        t8 = pd.concat([pd.DataFrame(rows), per_fam], ignore_index=True)
        tidy8, md8 = ev.build_paper_tables([("TABLE8_fidelity_plus", "Fidelity+ explanation results", t8, ctx)])
        run.results["_tidy_part"].append(tidy8)
        run._tables_md_part3 = md8
        out["n_alerts_evaluated"] = int(fid_df["sample_id"].nunique())
    except Exception as exc:
        out["fidelity_plus"] = {"status": "NOT RUN", "reason": f"{type(exc).__name__}: {exc}",
                                "requirement": "graph sequences + encode_with_attention (handoff contract §C/§E)"}
        run.fig_status["fig10_fidelity_plus.png"] = f"NOT RUN: {type(exc).__name__}: {exc}"
        print(f"[xai] Fidelity+ NOT RUN: {exc}", file=sys.stderr)
    run.results["xai"] = out
    _join_explanations_into_predictions(run, explanations)


def _join_explanations_into_predictions(run: Runner, explanations: List[Dict[str, Any]]) -> None:
    """One auditable table: per-alert attribution is joined onto results/predictions.csv (§30)."""
    if not explanations:
        return
    rows = [{"sample_id": str(e["alert_id"]).split(":")[-1],
             "top_host": (e["top_nodes"] or [{}])[0].get("host"),
             "top_host_attention": (e["top_nodes"] or [{}])[0].get("attention"),
             "top_flow_record": (e["top_edges"] or [{}])[0].get("flow_record"),
             "top_feature": (e["top_features"] or [{}])[0].get("feature"),
             "explanation_coverage_topk_nodes": e["explanation_coverage_topk_nodes"],
             "explanation_summary": e["summary"]} for e in explanations]
    add = pd.DataFrame(rows).drop_duplicates("sample_id", keep="first")
    path = run.path("results", "predictions.csv")
    if os.path.exists(path):
        pred = pd.read_csv(path)
        pred = pred.merge(add, on="sample_id", how="left")
        pred.to_csv(path, index=False)
        print(f"[out] {path}  (+{len(add.columns) - 1} explanation columns)")


def stage_ablation(run: Runner) -> None:
    base = run.cfg

    def factory(cfg: Dict[str, Any]) -> Member3Scorer:
        return run.build_scorer(cfg=cfg)

    records = run.records()
    scorer = run.load_scorer()
    test_pred = scorer.predict(records[records["split"] == SPLIT_TEST])
    out: Dict[str, Any] = {}
    strat = list(run.cfg.get("ablations", {}).get("threshold_strategies",
                                                  ["fixed_mean3sd", "fixed_percentile", "confidence_aware"]))
    out["threshold"] = ev.ablation_threshold(factory, base, records, strat)
    steps = list(run.cfg.get("ablations", {}).get("debounce_steps", [1, 2, 3, 4]))
    out["debounce"] = ev.ablation_debounce(scorer, base, test_pred, steps)
    out["memory"] = ev.ablation_memory(factory, base, records)
    if bool(run.cfg.get("ablations", {}).get("retrieval_timing", True)):
        out["retrieval"] = ev.ablation_retrieval(scorer, base, np.vstack(list(test_pred["embedding"])),
                                                n=int(run.cfg.get("ablations", {}).get("retrieval_queries", 300)))
    run.results["ablations"] = out

    t5 = pd.DataFrame(out["threshold"])
    t6 = pd.DataFrame(out["debounce"])
    t7 = pd.DataFrame(out["memory"])
    t9 = _ablation_summary(out, t5, t6, t7)
    ab_rows = pd.concat([t.assign(ablation=a) for a, t in
                         (("threshold", t5), ("debounce", t6), ("memory", t7),
                          ("retrieval", pd.DataFrame(out.get("retrieval", []))))], ignore_index=True)
    run.write_csv("ablations.csv", ab_rows)
    ctx = {"data_source": run.data_source, "config_id": run.config_id, "split": SPLIT_TEST}
    tidy9, md9 = ev.build_paper_tables([
        ("TABLE5_threshold_ablation", "Threshold strategy ablation", t5, ctx),
        ("TABLE6_debounce_ablation", "Multi-stage alert verification (debounce) ablation", t6, ctx),
        ("TABLE7_memory_adaptation", "Adaptive memory / online benign adaptation results", t7, ctx),
        ("TABLE9_ablation_summary", "Consolidated Member-3 ablation summary", t9, ctx)])
    run.results["_tidy_part"].append(tidy9)
    run._tables_md_part4 = md9

    fs = run.fig_status
    if len(t5):
        ev.figure_bar_ablation(t5, "strategy", ("f1", "precision", "recall", "auprc"),
                               "Figure 8: Threshold strategy ablation (sample-level)", "fig08_threshold_ablation.png",
                               run.figures_dir, fs, note_col="fpr_pct")
        ev.figure_bar_ablation(t5, "strategy", ("fpr_pct",), "Figure 8b: false-positive rate by threshold strategy (%)",
                               "fig08b_threshold_ablation_fpr.png", run.figures_dir, fs, ylab="FPR (%)",
                               ylim01=False)
    if len(t6):
        ev.figure_bar_ablation(t6, "debounce_steps", ("alert_f1", "alert_precision", "alert_recall"),
                               "Figure 9: multi-stage alert verification ablation (alert-level)",
                               "fig09_debounce_ablation.png", run.figures_dir, fs, note_col="alert_fpr_pct")
    if len(t7):
        ev.figure_bar_ablation(t7, "memory_mode", ("benign_fpr_before_adaptation", "benign_fpr_after_adaptation",
                                                  "test_f1"),
                               "Figure 7: memory adaptation ablation (benign FPR under drift + test F1)",
                               "fig07_memory_bank_ablation.png", run.figures_dir, fs, ylim01=False)
    ev.figure_cross_dataset(pd.DataFrame(run.results.get("evaluation", {}).get("per_dataset", [])),
                            run.figures_dir, fs)
    ev.figure_family_breakdown(pd.read_csv(run.path("results", "attack_family_results.csv"))
                               if os.path.exists(run.path("results", "attack_family_results.csv")) else None,
                               run.figures_dir, fs)


def _ablation_summary(out: Dict[str, Any], t5: pd.DataFrame, t6: pd.DataFrame, t7: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t, arm, key in ((t5, "threshold", "strategy"), (t6, "debounce", "debounce_steps"),
                        (t7, "memory", "memory_mode")):
        if t is None or len(t) == 0:
            rows.append({"arm": arm, "variant": ev.NOT_RUN, "status": "NOT RUN"})
            continue
        for _, r in t.iterrows():
            rows.append({"arm": arm, "variant": str(r.get(key)), "status": r.get("status"),
                         "n_samples": r.get("n_samples", r.get("n_queries", "")),
                         "f1": r.get("f1", r.get("sample_f1", r.get("test_f1"))),
                         "precision": r.get("precision", r.get("sample_precision", r.get("test_precision"))),
                         "recall": r.get("recall", r.get("sample_recall", r.get("test_recall"))),
                         "fpr": r.get("fpr", r.get("sample_fpr", r.get("test_fpr"))),
                         "auprc": r.get("auprc"), "roc_auc": r.get("roc_auc"), "threshold": r.get("threshold"),
                         "alert_f1": r.get("alert_f1"), "alert_precision": r.get("alert_precision"),
                         "alert_fpr": r.get("alert_fpr"), "n_alerts": r.get("n_alerts"),
                         "detection_rate": r.get("detection_rate"),
                         "mean_detection_delay_steps": r.get("mean_detection_delay_steps"),
                         "benign_fpr_before_adaptation": r.get("benign_fpr_before_adaptation"),
                         "benign_fpr_after_adaptation": r.get("benign_fpr_after_adaptation"),
                         "memory_growth": r.get("memory_growth"), "n_updates": r.get("n_updates"),
                         "mean_ms_per_query": r.get("mean_ms_per_query")})
    if "retrieval" in out:
        for r in out["retrieval"]:
            rows.append({"arm": "retrieval_backend", "variant": str(r.get("index_type")), "status": r.get("status"),
                         "n_samples": r.get("n_queries"), "mean_ms_per_query": r.get("mean_ms_per_query"),
                         "recall_at_k_vs_bruteforce": r.get("recall_at_k_vs_bruteforce")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# finalisation
# --------------------------------------------------------------------------- #
def finalise(run: Runner) -> None:
    parts = [p for p in run.results.get("_tidy_part", []) if len(p)]
    tidy = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(tidy):
        run.write_csv("paper_tables.csv", tidy)
    body = "\n\n".join([getattr(run, "_tables_md_part1", ""), getattr(run, "_tables_md_part2", ""),
                         getattr(run, "_tables_md_part3", ""), getattr(run, "_tables_md_part4", "")]).strip()
    header = ("# Member 3 - publication tables\n\n"
              f"data_source: `{run.data_source}`\n\n"
              "Every row carries `dataset`, `split`, `config_id`, `n_samples`, `data_source` and `notes`.\n"
              "`status = NOT RUN` means the measurement was blocked; the reason is in the row and in\n"
              "`results/manifest.json` - a blocked experiment is recorded, never dropped or estimated.\n")
    with open(run.path("results", "tables.md"), "w") as fh:
        fh.write(header + "\n" + body + "\n")
    ev.write_metrics(run.results, run.results_dir)
    run.fig_status["_manifest"] = "written"
    with open(run.path("figures", "figure_status.json"), "w") as fh:
        json.dump(run.fig_status, fh, indent=2)
    write_manifest(run, "final")
    print("\n=== stage summary ===")
    for st, info in run.stage_log.items():
        print(f"  {st:<12} {info['status']:<7} {info['seconds']:>7.2f}s" + (f"  {info['error']}" if info.get("error") else ""))
    print(f"  figures      {sum(1 for v in run.fig_status.values() if v == 'generated')} generated, "
          f"{sum(1 for v in run.fig_status.values() if str(v).startswith('NOT RUN'))} NOT RUN")
    failed = [s for s, i in run.stage_log.items() if i["status"] != "ok"]
    if failed:
        print(f"[warn] failed stages: {failed} (see results/manifest.json)", file=sys.stderr)
    return 1 if any(s in ("calibration", "evaluation") for s in failed) else 0


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Member 3 experiment runner")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args(argv)

    from member3.contract import load_config

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    run = Runner(cfg, args.config)
    stages = STAGES if args.stage == "all" else (args.stage,)
    fns: Dict[str, Callable[[Runner], None]] = {"calibration": stage_calibration, "evaluation": stage_evaluation,
                                               "xai": stage_xai, "ablation": stage_ablation}
    for st in stages:
        t0 = time.time()
        try:
            fns[st](run)
            run.update_manifest(st, True, time.time() - t0)
            print(f"[stage] {st} ok in {time.time() - t0:.1f}s")
        except Exception as exc:
            tb = traceback.format_exc(limit=4)
            print(f"[stage] {st} FAILED: {type(exc).__name__}: {exc}\n{tb}", file=sys.stderr)
            run.update_manifest(st, False, time.time() - t0, f"{type(exc).__name__}: {exc}")
            if args.fail_fast:
                finalise(run)
                return 2
    rc = finalise(run)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
