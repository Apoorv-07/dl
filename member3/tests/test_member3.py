"""Member-3 tests.

The research pipeline is exercised against upstream artefacts only; these tests therefore
construct their own tiny inputs in memory:

* ``TinyAdapter`` - a ~40-line deterministic stand-in for the *Member-2 interface* (encode /
  encode_with_attention / metadata / masking).  It is a test double for the interface, not a
  model, and none of its numbers are research results.
* ``make_graph`` / ``make_records`` - minimal temporal-graph sequences and a records table in the
  format `docs/MEMBER2_HANDOFF_CONTRACT.md` specifies, so the tests double as a check that the
  documented schema is actually loadable.

No dataset, no bundled fixture file, no committed outputs.  Run: ``pytest -q``.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from member3 import evaluate as ev  # noqa: E402
from member3.contract import (  # noqa: E402
    LABEL_ATTACK, LABEL_BENIGN, RECORD_COLUMNS, SPLIT_CALIBRATION, SPLIT_MEMORY, SPLIT_ONLINE_BENIGN,
    SPLIT_TEST, ContractError, assign_streams, assert_splits_clean, l2_normalize, load_graphs, mask_graph_sequence,
    save_graphs, save_records, load_records, validate_adapter, validate_records,
)
from member3.explain import build_explanation, bootstrap_ci, fidelity_plus  # noqa: E402
from member3.inference import (  # noqa: E402
    AlertTracker, Member3Scorer, calibrate_threshold, confidence_from_score,
)
from member3.memory import AdaptiveMemoryBank  # noqa: E402

DIM = 16
ACCEPTED_ACTIONS = ("inserted", "redundancy_replaced", "reservoir_replaced")
N_FEAT = 6
N_SNAP = 3
N_NODES = 5
N_EDGES = 7


# --------------------------------------------------------------------------- #
# test doubles for the upstream interface
# --------------------------------------------------------------------------- #
class TinyAdapter:
    """Deterministic stand-in implementing the Member-2 adapter interface (contract §A-C)."""

    def __init__(self, dim: int = DIM, seed: int = 0, n_features: int = N_FEAT, n_snapshots: int = N_SNAP):
        rng = np.random.default_rng(seed)
        self.dim, self.n_features, self.n_snapshots = int(dim), int(n_features), int(n_snapshots)
        d = 2 * n_features + 2
        self._proj = rng.standard_normal((d, self.dim)) / np.sqrt(d)

    def metadata(self):
        return {"architecture_id": "tiny-test-double", "checkpoint_version": "test", "framework": "numpy",
                "framework_version": np.__version__, "embedding_dim": self.dim, "device": "cpu",
                "l2_normalized": True, "post_projection_head": True, "n_features": self.n_features,
                "n_snapshots": self.n_snapshots, "attention_available": True, "masking_available": True,
                "trainable": False, "checkpoint_sha256": "0" * 16, "preprocessing_id": "test"}

    @staticmethod
    def _summary(seq):
        stats = []
        for snap in seq["snapshots"]:
            x = np.asarray(snap["x"], dtype=np.float64)
            ei = np.asarray(snap["edge_index"], dtype=np.int64).reshape(2, -1)
            ea = np.asarray(snap["edge_attr"], dtype=np.float64)
            agg = np.zeros((x.shape[0], ea.shape[1] if ea.size else 1))
            cnt = np.zeros((x.shape[0], 1))
            if ei.shape[1]:
                np.add.at(agg, ei[1], ea)
                np.add.at(cnt, ei[1], 1.0)
            agg = agg / np.maximum(cnt, 1.0)
            stats.append(np.concatenate([agg.mean(axis=0), agg.std(axis=0),
                                        [ei.shape[1] / max(x.shape[0], 1), float(x.mean())]]))
        return np.mean(stats, axis=0)

    def encode(self, graph_sequence, masked_nodes=None, masked_edges=None):
        if masked_nodes or masked_edges:
            graph_sequence = mask_graph_sequence(graph_sequence, masked_nodes, masked_edges)
        z = self._proj.T @ self._summary(graph_sequence)
        return (z / (np.linalg.norm(z) + 1e-12)).astype(np.float32)

    def encode_batch(self, seqs):
        return np.vstack([self.encode(s) for s in seqs])

    def encode_with_attention(self, graph_sequence):
        snaps = graph_sequence["snapshots"]
        n = len(np.asarray(snaps[0]["node_ids"]))
        na = np.zeros(n)
        weights = []
        for snap in snaps:
            ei = np.asarray(snap["edge_index"], dtype=np.int64).reshape(2, -1)
            ea = np.asarray(snap["edge_attr"], dtype=np.float64)
            w = np.abs(ea).sum(axis=1) if ea.size else np.zeros(ei.shape[1])
            weights.append(w)
            if ei.shape[1]:
                np.add.at(na, ei[1], w)
        edge_attn = np.mean([np.pad(w, (0, max(0, N_EDGES - len(w)))) for w in weights], axis=0)
        na = np.exp(na - na.max())
        na = na / na.sum()
        gmeta = {"node_ids": [list(s.get("node_ids", [])) for s in snaps],
                 "edge_record_ids": [list(s.get("edge_record_ids", [])) for s in snaps],
                 "snapshot_ids": list(graph_sequence.get("snapshot_ids", range(len(snaps)))),
                 "timestamps": list(graph_sequence.get("timestamps", [])),
                 "graph_ids": list(graph_sequence.get("graph_ids", range(len(snaps)))),
                 "feature_names": list(graph_sequence.get("feature_names", [])),
                 "primary_snapshot": 0}
        return {"embedding": self.encode(graph_sequence), "node_attention": na.astype(np.float32),
                "edge_attention": edge_attn.astype(np.float32), "graph_metadata": gmeta}

    def mask_nodes(self, seq, node_ids):
        return mask_graph_sequence(seq, node_ids=node_ids)

    def mask_edges(self, seq, edge_ids):
        return mask_graph_sequence(seq, edge_ids=edge_ids)


ADAPTER_SOURCE = textwrap.dedent('''
    """Adapter handed over by Member 2 (test copy: same entry points as the real file)."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from tests.test_member3 import TinyAdapter

    class FoundationModelAdapter(TinyAdapter):
        @classmethod
        def from_checkpoint(cls, ckpt, model_config):
            return cls()

    def build_adapter(checkpoint=None, model_config=None, config=None):
        return FoundationModelAdapter()
''')


def make_graph(sid: str, seed: int = 0, boost=None, mag: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    snaps = []
    for k in range(N_SNAP):
        src = rng.integers(0, N_NODES, N_EDGES)
        dst = rng.integers(0, N_NODES, N_EDGES)
        ea = np.abs(rng.normal(0.4, 0.2, (N_EDGES, N_FEAT)))
        if boost is not None:
            ea[:, list(boost)] += mag
        snaps.append({"x": rng.normal(0, 0.4, (N_NODES, N_FEAT)).tolist(),
                      "edge_index": np.vstack([src, dst]).tolist(),
                      "edge_attr": ea.tolist(),
                      "node_ids": [f"10.0.{k}.{i}" for i in range(N_NODES)],
                      "edge_record_ids": [f"{sid}-s{k}-e{j}" for j in range(N_EDGES)]})
    return {"sample_id": sid, "snapshots": snaps,
            "snapshot_ids": [f"{sid}-t{k}" for k in range(N_SNAP)],
            "timestamps": [float(k * 30) for k in range(N_SNAP)],
            "graph_ids": [f"{sid}-g{k}" for k in range(N_SNAP)],
            "feature_names": [f"nf_f{i:02d}" for i in range(N_FEAT)]}


def raw_cfg(dim: int = DIM, **over) -> dict:
    cfg = {
        "seed": 7,
        "member2": {"embedding_dim": dim, "n_features": N_FEAT, "n_snapshots": N_SNAP,
                    "artifacts_dir": "does-not-exist", "adapter_path": None},
        "memory": {"capacity": 48, "index_type": "numpy", "hnsw_M": 8, "ef_construction": 40,
                   "ef_search": 40, "selection": "kcenter_greedy", "replacement": "redundant"},
        "retrieval": {"k": 4, "neighbor_aggregation": "max"},
        "threshold": {"strategy": "confidence_aware", "percentile": 95.0, "k_mad": 3.0, "safety_margin": 0.1,
                      "min_calibration_samples": 8, "confidence_sharpness": 1.0},
        "alerting": {"debounce_steps": 3},
        "adaptation": {"enabled": True, "benign_confidence_max": 0.05},
        "xai": {"mask_target": "nodes", "mask_top_k": 1, "top_nodes": 2, "top_edges": 2, "top_features": 3,
                "max_alerts": 6, "methods": ["attention", "random", "lowest_attention"]},
        "statistics": {"bootstrap_iterations": 60, "repeats": 1},
        "evaluation": {"train_dataset": "DS-TRAIN", "test_datasets": ["DS-TEST"]},
        "data": {"window_seconds": 30.0, "records_path": None, "graph_path": None, "dataset_label": "DS-TEST"},
        "benchmark": {"n_samples": 12, "repeats": 1, "warmup": 1, "latency_target_ms": 5.0},
        "output": {"results_dir": "results", "figures_dir": "figures", "state_dir": "state"},
        "ablations": {"threshold_strategies": ["confidence_aware", "fixed_mean3sd", "percentile_only"]},
    }
    for k, v in over.items():
        cfg[k] = v
    return cfg


def make_records(cfg=None, n_train=48, n_cal=32, noise=0.02, attack_noise=0.02,
                  families=("Exploits", "Reconnaissance"), burst=5, gap=4,
                  train_dataset="DS-TRAIN", test_dataset="DS-TEST", seed=1):
    """Records + graph sequences in the documented upstream format.

    Benign windows sit in one regime on the unit sphere, each attack family in an orthogonal
    regime; the gap is large by construction so the assertions below are about Member-3
    machinery (ordering, gating, counting), not about detection difficulty.
    """
    cfg = cfg or raw_cfg()
    rng = np.random.default_rng(seed)
    benign = np.zeros(DIM); benign[0] = 1.0
    rows, graphs = [], {}
    counter = [0]

    def emit(split, dataset, label, family, z, boost=None, mag=0.0, episode="benign"):
        k = counter[0]
        counter[0] += 1
        sid = {"train_benign": "tra", "calibration_benign": "cal", "test": "tes",
               "online_benign": "onl"}[split] + f"-{k:06d}"
        graphs[sid] = make_graph(sid, seed=k, boost=boost, mag=mag)
        rows.append({"dataset_id": dataset, "split": split, "sample_id": sid, "window_id": f"w{k:06d}",
                     "timestamp": float(k),
                     "embedding": l2_normalize(z + noise * rng.standard_normal(DIM)).astype(np.float32),
                     "label": label, "attack_family": family, "episode_id": episode})

    for _ in range(n_train):
        emit(SPLIT_MEMORY, train_dataset, LABEL_BENIGN, "benign", benign)
    for _ in range(n_cal):
        emit(SPLIT_CALIBRATION, train_dataset, LABEL_BENIGN, "benign", benign)

    # test stream: benign background, one contiguous attack burst per family (an "episode")
    stream = [(LABEL_BENIGN, "benign", None)] * gap
    for fi, fam in enumerate(families):
        stream += [(LABEL_ATTACK, fam, f"E{fi}")] * burst
        stream += [(LABEL_BENIGN, "benign", None)] * gap
    for label, fam, episode in stream:
        if label == LABEL_ATTACK:
            center = np.zeros(DIM)
            center[1 + list(families).index(fam)] = 1.0
            emit(SPLIT_TEST, test_dataset, LABEL_ATTACK, fam, center, boost=[0, 1], mag=6.0,
                 episode=episode)
        else:
            emit(SPLIT_TEST, test_dataset, LABEL_BENIGN, "benign", benign)

    # online benign stream: same regime, slightly noisier (drift the memory may absorb)
    for i in range(n_cal):
        emit(SPLIT_ONLINE_BENIGN, test_dataset, LABEL_BENIGN, "benign", benign + 0.25 * rng.standard_normal(DIM))

    df = pd.DataFrame(rows)
    df["stream_id"] = df["dataset_id"] + ":" + df["split"]
    df = df.sort_values(["stream_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["stream_pos"] = df.groupby("stream_id").cumcount()
    ep = []
    for g in df[df["split"] == SPLIT_TEST].groupby(df["label"].ne("benign").cumsum()):
        pass
    # one episode per contiguous attack burst (this is exactly what Member 1's episode_id means)
    eid, run_id, prev = [], None, None
    for _, row in df.iterrows():
        if row["label"] == LABEL_ATTACK:
            if prev != (row["dataset_id"], row["attack_family"], True):
                run_id = f"{row['attack_family']}#{len(set(run for run in eid if run))}"
            eid.append(run_id)
            prev = (row["dataset_id"], row["attack_family"], True)
        else:
            eid.append("benign")
            prev = None
    df["episode_id"] = eid
    validate_records(df, expected_dim=DIM)
    assert_splits_clean(df, permitted_reference_datasets=[train_dataset])
    return df, graphs


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A complete Member-3 scorer fitted on the tiny records, plus the artefacts on disk."""
    d = tmp_path_factory.mktemp("m3")
    cfg = raw_cfg()
    rec, graphs = make_records(cfg)
    art = d / "member2_artifacts"
    art.mkdir()
    (art / "model_adapter.py").write_text(ADAPTER_SOURCE)
    (art / "checkpoint.pt").write_bytes(b"tiny-test-checkpoint")
    save_records(rec, str(d / "records.npz"))
    save_graphs(graphs, str(d / "graphs.json"))
    cfg["member2"]["artifacts_dir"] = str(art)
    cfg["data"]["records_path"] = str(d / "records.npz")
    cfg["data"]["graph_path"] = str(d / "graphs.json")
    from member3.contract import load_adapter
    adapter, meta = load_adapter(cfg)
    scorer = Member3Scorer.build(cfg, adapter, meta, rec)
    return {"dir": str(d), "cfg": cfg, "records": rec, "graphs": graphs, "scorer": scorer,
            "adapter": adapter, "adapter_meta": meta}


# --------------------------------------------------------------------------- #
# adaptive memory bank
# --------------------------------------------------------------------------- #
def test_memory_capacity_and_retrieval_bounds():
    rng = np.random.default_rng(0)
    bank = AdaptiveMemoryBank(dim=DIM, capacity=8, index_type="numpy", k=2, seed=0)
    X = l2_normalize(rng.standard_normal((25, DIM)))
    bank.fit(X, [{"sample_id": f"s{i}"} for i in range(25)])
    assert bank.stats.n == 8 and bank.stats.dim == DIM
    sims, idx = bank.search(X[:1])
    assert sims.shape == (1, 2) and np.all(sims <= 1.0 + 1e-9) and idx.max() < 8
    rec = bank.add(X[9], {"sample_id": "late"})
    assert rec["action"] in ACCEPTED_ACTIONS + ("dropped_redundant",)
    assert bank.stats.n == 8                      # capacity is a hard bound


def test_memory_selection_is_deterministic_and_reloadable(tmp_path):
    rng = np.random.default_rng(3)
    X = l2_normalize(rng.standard_normal((40, DIM)))
    a = AdaptiveMemoryBank(dim=DIM, capacity=10, index_type="numpy", k=3, seed=11).fit(X)
    b = AdaptiveMemoryBank(dim=DIM, capacity=10, index_type="numpy", k=3, seed=11).fit(X)
    assert np.array_equal(a.vectors().astype("float32"), b.vectors().astype("float32"))
    base = os.fspath(tmp_path / "mem")
    a.save(base)
    c = AdaptiveMemoryBank.load(base + ".bin")
    assert np.allclose(c.vectors(), a.vectors()) and c.stats.version == a.stats.version
    sa, _ = a.search(X[:5])
    sc, _ = c.search(X[:5])
    assert np.allclose(sa, sc, atol=1e-6)


def test_attack_samples_can_never_enter_memory():
    rng = np.random.default_rng(5)
    bank = AdaptiveMemoryBank(dim=DIM, capacity=10, index_type="numpy", k=2, seed=0)
    bank.fit(l2_normalize(rng.standard_normal((5, DIM))))
    n_before = bank.stats.n
    z = l2_normalize(rng.standard_normal(DIM))
    with pytest.raises(PermissionError, match="MEMORY CORRUPTION BLOCKED"):
        bank.update_if_benign(z, confidence=0.99, score=0.9, threshold=1.0,
                              metadata={"sample_id": "x", "label": LABEL_ATTACK})
    assert bank.stats.n == n_before and bank.stats.n_rejected_attack == 1
    # above-threshold and merely-uncertain samples are refused (logged, not raised)
    r1 = bank.update_if_benign(z, confidence=0.0, score=0.8, threshold=0.5, metadata={"label": LABEL_BENIGN})
    r2 = bank.update_if_benign(z, confidence=0.6, score=0.1, threshold=0.5, metadata={"label": LABEL_BENIGN})
    assert r1["action"] == "rejected_above_threshold" and r2["action"] == "rejected_uncertain"
    assert bank.stats.n == n_before
    r3 = bank.update_if_benign(z, confidence=0.0, score=0.1, threshold=0.5, metadata={"label": LABEL_BENIGN})
    assert r3["action"] in ACCEPTED_ACTIONS
    assert bank.stats.n == n_before + 1


def test_immutable_bank_refuses_writes():
    rng = np.random.default_rng(6)
    bank = AdaptiveMemoryBank(dim=DIM, capacity=6, index_type="numpy", k=2, seed=0)
    bank.fit(l2_normalize(rng.standard_normal((6, DIM))))
    bank.set_immutable(True)
    with pytest.raises(PermissionError, match="immutable"):
        bank.add(l2_normalize(rng.standard_normal(DIM)))


@pytest.mark.skipif(not os.environ.get("MEMBER3_TEST_FAISS", "1") == "1", reason="faiss disabled")
def test_faiss_index_matches_exact_search():
    try:
        import faiss  # noqa: F401
    except Exception:
        pytest.skip("faiss not installed")
    rng = np.random.default_rng(8)
    X = l2_normalize(rng.standard_normal((200, DIM)))
    q = l2_normalize(rng.standard_normal((20, DIM)))
    ref = AdaptiveMemoryBank(dim=DIM, capacity=200, index_type="numpy", k=5, seed=0).fit(X)
    out = {}
    for kind in ("hnsw", "flat"):
        bank = AdaptiveMemoryBank(dim=DIM, capacity=200, index_type=kind, k=5, hnsw_M=32,
                                  ef_construction=200, ef_search=128, seed=0).fit(X)
        sims, idx = bank.search(q)
        exact_sims, exact_idx = ref.search(q)
        out[kind] = (sims, idx, exact_idx)
    assert out["flat"][1].tolist() == out["flat"][2].tolist()          # exact index == numpy
    recall = np.mean([len(set(a) & set(b)) / 5.0 for a, b in zip(out["hnsw"][1], out["hnsw"][2])])
    assert recall >= 0.9, f"HNSW recall@5 too low: {recall}"           # approximate, but must be near-exact
    assert np.all(out["hnsw"][0] <= 1.0 + 1e-6)


# --------------------------------------------------------------------------- #
# score + threshold + confidence
# --------------------------------------------------------------------------- #
def test_anomaly_score_is_one_minus_cosine_similarity(built):
    sc = built["scorer"]
    z = l2_normalize(np.eye(DIM)[0])
    res = sc.score_embeddings(np.vstack([z]))
    assert res["anomaly_score"].shape == (1,)
    assert 0.0 <= res["anomaly_score"][0] <= 2.0
    assert np.isclose(res["anomaly_score"][0], 1.0 - res["similarity"][0])
    # a vector stored in the memory must score ~0 (same definition, not a second one)
    mem = sc.memory.vectors()[:1]
    assert sc.score_embeddings(mem)["anomaly_score"][0] < 1e-5


def test_threshold_uses_calibration_split_only_and_is_conservative(built):
    sc, rec, cfg = built["scorer"], built["records"], built["cfg"]
    cal = rec[rec["split"] == SPLIT_CALIBRATION]
    cal_scores = sc.score_embeddings(np.vstack(list(cal["embedding"])))["anomaly_score"]
    tau = float(sc.threshold.value)
    pctl = float(np.percentile(cal_scores, cfg["threshold"]["percentile"]))
    assert tau > pctl                                     # MAD band + safety margin are added on top
    assert float(sc.threshold.median) <= pctl < tau
    assert sc.threshold.n_calibration == len(cal) and sc.threshold.calibrated_on.startswith(SPLIT_CALIBRATION)
    assert set(cal["label"]) == {LABEL_BENIGN}
    assert SPLIT_TEST not in sc.threshold.calibrated_on


def test_threshold_refuses_tiny_or_attacked_calibration():
    cfg = raw_cfg()
    rng = np.random.default_rng(2)
    tiny = np.abs(rng.random(5)) * 0.1
    with pytest.raises(ContractError, match="too small"):
        calibrate_threshold(tiny, cfg, calibrated_on="calibration_benign", split_tags=[SPLIT_CALIBRATION])
    bad = dict(cfg["threshold"])
    bad["min_calibration_samples"] = 3
    y = np.concatenate([np.abs(rng.random(20)) * 0.1, np.full(10, 0.9)])
    with pytest.raises(PermissionError, match="THRESHOLD CONTAMINATION BLOCKED"):
        calibrate_threshold(y, {**cfg, "threshold": bad}, calibrated_on="test", split_tags=[SPLIT_TEST])
    with pytest.raises(ContractError, match="non-finite"):
        calibrate_threshold(np.array([0.1, np.nan, 0.2] * 12), {**cfg, "threshold": bad},
                            calibrated_on="calibration_benign", split_tags=[SPLIT_CALIBRATION])


def test_confidence_is_monotone_in_the_score():
    tau, scale = 0.1, 0.02
    s = np.array([0.0, 0.05, 0.1, 0.2, 1.0])
    c = confidence_from_score(s, tau, scale, sharpness=1.0)
    assert np.all(np.diff(c) > 0) and c.min() >= 0.0 and c.max() <= 1.0
    assert np.isclose(c[2], 0.5, atol=1e-6)              # at the threshold: 50 %
    two = confidence_from_score(np.array([0.1, 1.0]), tau, scale)
    assert two[1] > two[0]


# --------------------------------------------------------------------------- #
# alert verification (3-step confirmation)
# --------------------------------------------------------------------------- #
def test_debounce_needs_three_consecutive_windows():
    tr = AlertTracker(debounce=3, window_seconds=30.0)
    states = [tr.update(1.0, 0.5, pos=i)["alert_state"] for i in range(3)]
    assert states == ["candidate", "candidate", "confirmed"]
    out = tr.update(0.0, 0.5, pos=3)
    assert out["alert_state"] == "none" and out["consecutive_anomalies"] == 0      # benign resets
    tr2 = AlertTracker(debounce=3)
    for i in range(3):
        assert tr2.update(1.0, 0.5, pos=i * 2)["alert_state"] != "confirmed"        # non-consecutive gaps


def test_verify_alerts_delay_and_rising_edge(built):
    sc = built["scorer"]
    pred = sc.predict(built["records"])
    pred = sc.verify_alerts(pred, window_seconds=30.0)
    test = pred[pred["split"] == SPLIT_TEST]
    att = test[test["label"] == LABEL_ATTACK]
    raised = test[test["alert_raised"]]
    n_ep = test.loc[test["episode_id"] != "benign", "episode_id"].nunique()
    assert n_ep == 2 and len(raised) == n_ep                       # one alert per attack burst and sorted(test.loc[test["alert_raised"], "episode_id"].unique()) == ["E0", "E1"]
    assert set(att["alert_state"]) <= {"candidate", "confirmed"}
    assert (att["consecutive_anomalies"] >= 1).all()
    delay = raised["detection_delay_steps"].dropna().to_numpy()
    assert len(delay) and np.all(delay == 2)                                  # confirmed on the 3rd window
    assert np.all(raised["detection_delay_seconds"] == 60.0)
    ben = test[test["label"] == LABEL_BENIGN]
    assert not ben["alert_raised"].any()                     # benign background between bursts: no alerts
    em = ev.episode_metrics(test, window_seconds=30.0)
    assert em["n_episodes"] == 2 and em["n_true_alerts"] == 2 and em["n_missed_episodes"] == 0
    assert em["n_false_alerts"] == 0 and em["detection_rate"] == pytest.approx(1.0)
    # the online stream is deliberately drifted: any alert there is a false alarm, and it is reported, not hidden
    assert int(pred[(pred["split"] == SPLIT_ONLINE_BENIGN) & pred["alert_raised"]].shape[0]) >= 0


def test_confirmed_alerts_carry_the_streak_and_memory_version(built):
    sc = built["scorer"]
    pred = sc.verify_alerts(sc.predict(built["records"]))
    conf = pred[pred["confirmed_alert"]]
    assert len(conf) and set(conf["consecutive_anomalies"]) >= {3}
    assert conf["memory_version"].nunique() == 1 and (conf["memory_version"] >= 0).all()
    assert np.all(conf["anomaly_score"].to_numpy() > conf["threshold"].to_numpy())
    assert {"neighbor_sample_ids", "neighbor_similarities", "nearest_normal_similarity", "checkpoint_version",
            "dataset_split", "threshold_method"} <= set(pred.columns)   # every prediction stays traceable
    assert conf["neighbor_sample_ids"].str.len().gt(0).all()
    assert np.allclose(conf["anomaly_score"].to_numpy() + conf["nearest_normal_similarity"].to_numpy(), 1.0,
                       atol=1e-6)                          # score = 1 - similarity, one definition


# --------------------------------------------------------------------------- #
# online memory adaptation (memory, never the model)
# --------------------------------------------------------------------------- #
def test_online_adaptation_gate_admits_only_in_regime_benign(built):
    """Two directions of the same gate: drifted benign windows are refused (and counted);
    in-regime benign windows are admitted.  Model weights never move in either case."""
    import copy
    sc = copy.deepcopy(built["scorer"])
    sc.memory.set_immutable(False)
    sc.cfg = {**sc.cfg, "adaptation": {"enabled": True, "benign_confidence_max": 0.8}}
    n0, v0 = sc.memory.stats.n, sc.memory.stats.version
    before = sc.memory.vectors().copy()
    online = built["records"][built["records"]["split"] == SPLIT_ONLINE_BENIGN]
    n_online = len(online)

    res = sc.adapt_online(built["records"])                      # drifted stream, score > tau
    assert res["status"] == "ok" and res["n_candidates"] == n_online
    assert sum(res["actions"].values()) == n_online and res["n_updates"] == 0
    assert res["actions"] == {"rejected_above_threshold": n_online}      # nothing slipped in
    assert res["n_rejected"] == n_online and res["model_weights_updated"] is False
    assert np.array_equal(before, sc.memory.vectors()) and sc.memory.stats.version == v0
    assert sc.memory.stats.immutable is True                     # mutability restored, not left open

    # in-regime benign windows (same stream, embeddings pulled back onto the normal manifold)
    rng = np.random.default_rng(21)
    base = l2_normalize(np.eye(sc.memory.stats.dim)[0] + 0.004 * rng.standard_normal((n_online, DIM)))
    ok = online.copy()
    ok["embedding"] = [b.astype(np.float32) for b in base]
    res2 = sc.adapt_online(ok)
    assert res2["n_updates"] > 0 and not (set(res2["actions"]) & {"rejected_above_threshold",
                                                                   "rejected_uncertain", "rejected_attack_label"})
    assert set(res2["actions"]) <= {"inserted", "redundancy_replaced", "reservoir_replaced", "dropped_redundant"}
    assert res2["n_rejected"] == res2["actions"].get("dropped_redundant", 0)   # redundancy policy, not the gate
    assert res2["memory_version_after"] > v0 and not np.array_equal(before, sc.memory.vectors())
    assert res2["memory_size_after"] <= built["cfg"]["memory"]["capacity"]
    ids = {e["sample_id"] for e in sc.memory.entries() if e.get("insert_stage") == "online_update"}
    assert ids and ids <= set(online["sample_id"])               # only the online stream fed the bank


def test_adaptation_refused_when_disabled(built):
    import copy
    sc = copy.deepcopy(built["scorer"])
    sc.cfg = {**sc.cfg, "adaptation": {"enabled": False}}
    sc.memory.set_immutable(False)
    n0 = sc.memory.stats.n
    res = sc.adapt_online(built["records"])
    assert res["status"] == "disabled" and res["n_candidates"] > 0 and sc.memory.stats.n == n0


# --------------------------------------------------------------------------- #
# graph attribution + Fidelity+
# --------------------------------------------------------------------------- #
def test_masking_helper_drops_nodes_and_edges():
    g = make_graph("g0", seed=0)
    n_before = len(g["snapshots"][0]["x"])
    masked = mask_graph_sequence(g, node_ids=[1])
    assert len(masked["snapshots"][0]["x"]) == n_before - 1
    assert all(int(e) < n_before - 1 for e in np.asarray(masked["snapshots"][0]["edge_index"]).ravel())
    e_masked = mask_graph_sequence(g, edge_ids=[0])
    assert np.allclose(np.asarray(e_masked["snapshots"][0]["edge_attr"])[0], 0.0)
    with pytest.raises(ContractError, match="nothing to mask"):
        mask_graph_sequence(g)


def test_explanation_object_is_auditable(built):
    sc, graphs = built["scorer"], built["graphs"]
    sid = [s for s in graphs if s.startswith("tes") and
           built["records"].set_index("sample_id").loc[s, "label"] == LABEL_ATTACK][0]
    row = sc.predict(built["records"].set_index("sample_id").loc[[sid]].reset_index()).iloc[0]
    expl = build_explanation(built["adapter"], graphs[sid], float(row["anomaly_score"]), float(row["threshold"]),
                             float(row["confidence"]), alert_id=f"al:{sid}", k_nodes=2, k_edges=2, k_features=3)
    assert expl["alert_id"] == f"al:{sid}"
    assert len(expl["top_nodes"]) == 2 and len(expl["top_edges"]) == 2 and len(expl["top_features"]) == 3
    assert all({"node_index", "host", "attention", "attention_share"} <= set(n) for n in expl["top_nodes"])
    assert all(str(n["host"]).startswith("10.0.") for n in expl["top_nodes"])          # real host ids
    assert all({"edge_index", "flow_record", "attention"} <= set(e0) for e0 in expl["top_edges"])
    assert all({"feature", "attribution_weight"} <= set(f0) for f0 in expl["top_features"])
    assert expl["feature_attribution_source"] and expl["n_snapshots"] == N_SNAP
    assert expl["exceeds_threshold"] is True and expl["anomaly_score"] == pytest.approx(
        round(float(row["anomaly_score"]), 6), abs=1e-6)
    assert expl["top_nodes"][0]["attention"] >= expl["top_nodes"][1]["attention"]      # ranked
    assert 0.0 <= expl["explanation_coverage_topk_nodes"] <= 1.0
    assert set(json.loads(json.dumps(expl))) >= {"alert_id", "top_nodes", "top_edges", "top_features",
                                                 "summary", "anomaly_score", "threshold", "confidence",
                                                 "temporal_snapshot", "memory_source_dataset"}
    assert sid in expl["summary"] or f"{float(row['anomaly_score']):.3f}" in expl["summary"]
    assert expl["anomaly_score"] == pytest.approx(float(row["anomaly_score"]), abs=1e-6)
    assert expl["threshold"] == pytest.approx(float(row["threshold"]), abs=1e-6)



def test_malformed_attention_payload_is_rejected_not_propagated():
    """A mis-sized attention vector upstream must become a contract error naming the field."""
    from member3.explain import validate_attention_payload

    g = make_graph("g0", seed=1)
    good = TinyAdapter().encode_with_attention(g)
    assert validate_attention_payload(good, g, DIM)["node_attention"].size <= N_NODES
    for bad, why in [({**good, "node_attention": np.ones(N_NODES + 4)}, "node_attention"),
                     ({**good, "node_attention": np.array([])}, "node_attention"),
                     ({**good, "node_attention": np.array([np.nan] * N_NODES)}, "non-finite"),
                     ({**good, "node_attention": np.full(N_NODES, 3.0)}, "softmax-normalised"),
                     ({**good, "embedding": np.zeros(DIM + 1)}, "embedding has"),
                     ({**good, "edge_attention": np.ones(N_EDGES * 3)}, "edge_attention"),
                     ({k: v for k, v in good.items() if k != "node_attention"}, "missing"),
                     (np.zeros(DIM), "must return a dict")]:
        with pytest.raises(ContractError, match=why):
            validate_attention_payload(bad, g, DIM)
    with pytest.raises(ContractError, match="ids and"):
        payload = {**good, "graph_metadata": {**good["graph_metadata"],
                                              "node_ids": [[f"bogus.{i}" for i in range(N_NODES + 2)] * N_SNAP]}}
        validate_attention_payload(payload, g, DIM)
    # dim<=0 means "unknown": the embedding check is skipped rather than falsely tripped
    assert validate_attention_payload(good, g, 0) is not None
    with pytest.raises(ContractError, match="node_attention"):
        build_explanation(_BrokenAttentionAdapter(), g, 0.5, 0.1, 0.9, "al:1")


class _BrokenAttentionAdapter(TinyAdapter):
    def encode_with_attention(self, seq):
        out = super().encode_with_attention(seq)
        return {**out, "node_attention": np.ones(N_NODES + 3, dtype=np.float32)}


def test_fidelity_plus_recomputes_through_the_adapter(built):
    sc, graphs, cfg = built["scorer"], built["graphs"], built["cfg"]
    pred = sc.verify_alerts(sc.predict(built["records"]))
    alerts = pred[(pred["alert_raised"]) & (pred["split"] == SPLIT_TEST)]
    was_immutable = bool(sc.memory.stats.immutable)
    df, summary = fidelity_plus(sc, alerts, graphs, {**cfg, "xai": {**cfg["xai"], "max_alerts": 2}})
    assert {"sample_id", "masking_method", "mask_target", "n_components_masked", "anomaly_score_before",
            "anomaly_score_after", "score_reduction", "fidelity_plus_pct", "attention_coverage"} <= set(df.columns)
    assert len(df) == 2 * len(cfg["xai"]["methods"]) and set(df["masking_method"]) == set(cfg["xai"]["methods"])
    assert (df["n_components_masked"] == 1).all() and (df["mask_target"] == "nodes").all()
    assert summary["_meta"]["n_samples_without_graph"] == 0
    assert set(summary) - {"_meta"} == set(cfg["xai"]["methods"])
    for m in cfg["xai"]["methods"]:
        assert 0.0 <= summary[m]["frac_score_drops"] <= 1.0 and summary[m]["n_alerts"] == 2
        assert summary[m]["ci95_low"] <= summary[m]["mean"] <= summary[m]["ci95_high"]
    # masking a boosted graph must move the embedding: the recompute is real, not a copy
    assert (df["anomaly_score_before"] != df["anomaly_score_after"]).any()
    assert bool(sc.memory.stats.immutable) == was_immutable    # frozen during, restored after
    assert "cosine" in summary["_meta"]["formula"] and "eps" in summary["_meta"]
    with pytest.raises(RuntimeError, match=r"Fidelity\+ NOT RUN"):
        fidelity_plus(sc, alerts, {}, {**cfg})          # no graphs handed over -> loud, not silent


# --------------------------------------------------------------------------- #
# metrics, episodes, families, zero-day protocol
# --------------------------------------------------------------------------- #
def test_classification_metrics_are_exact_on_a_known_table():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([1, 0, 0, 1, 1, 0])
    m = ev.classification_metrics(y, p, scores=np.array([0.9, 0.2, 0.1, 0.6, 0.8, 0.05]))
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3) and m["fpr"] == pytest.approx(1 / 3)
    assert m["fnr"] == pytest.approx(1 / 3) and m["n_samples"] == 6 and m["n_positive"] == 3
    assert m["confusion"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert m["metric_validity"] == "ok"


def test_episode_metrics_counts_detected_missed_and_false():
    rows = []
    for i in range(9):
        rows.append({"label": LABEL_ATTACK, "episode_id": "E1", "stream_id": "s", "stream_pos": i,
                     "alert_raised": i == 2, "confirmed_alert": i >= 2, "detection_delay_steps": 2,
                     "detection_delay_seconds": 60.0, "anomaly_score": 0.9, "threshold": 0.1,
                     "confidence": 0.99, "consecutive_anomalies": min(i + 1, 3)})
    for i in range(9, 14):
        rows.append({"label": LABEL_ATTACK, "episode_id": "E2", "stream_id": "s", "stream_pos": i,
                     "alert_raised": False, "confirmed_alert": False, "detection_delay_steps": np.nan,
                     "detection_delay_seconds": np.nan, "anomaly_score": 0.2, "threshold": 0.5,
                     "confidence": 0.1, "consecutive_anomalies": 0})
    rows.append({"label": LABEL_BENIGN, "episode_id": "benign", "stream_id": "s", "stream_pos": 14,
                 "alert_raised": True, "confirmed_alert": True, "detection_delay_steps": 0,
                 "detection_delay_seconds": 0.0, "anomaly_score": 0.6, "threshold": 0.5,
                 "confidence": 0.9, "consecutive_anomalies": 3})
    out = ev.episode_metrics(pd.DataFrame(rows), window_seconds=30.0)
    assert out["n_episodes"] == 2 and out["n_true_alerts"] == 1 and out["n_missed_episodes"] == 1
    assert out["n_false_alerts"] == 1 and out["detection_rate"] == pytest.approx(0.5)
    assert out["mean_detection_delay_steps"] == pytest.approx(2.0)
    assert out["mean_detection_delay_seconds"] == pytest.approx(60.0)
    assert "no episode_id" in ev.episode_metrics(pd.DataFrame({"label": []}))["metric_validity"]


def burst_count(rec, fam, n=5):
    return int(((rec["attack_family"] == fam) & (rec["split"] == SPLIT_TEST)).sum()) if fam != "benign" \
        else int(((rec["label"] == LABEL_BENIGN) & (rec["split"] == SPLIT_TEST)).sum())


def test_per_attack_family_is_family_versus_benign(built):
    sc = built["scorer"]
    pred = sc.predict(built["records"])
    pred = sc.verify_alerts(pred)
    fam = ev.per_attack_family(pred, built["cfg"])
    assert set(fam["attack_family"]) >= {"Exploits", "Reconnaissance"}
    for _, r in fam.iterrows():
        assert r["n_positive"] == (burst_count(built["records"], r["attack_family"])
                                   if r["attack_family"] != "benign" else len(built["records"].query(
                                       "split == 'test' and label == 'benign'")))
        if r["attack_family"] != "benign":
            assert r["recall"] == pytest.approx(1.0) and r["precision"] == pytest.approx(1.0)
            assert r["f1"] == pytest.approx(1.0) and r["n_negative"] > 0


def test_leave_one_family_out_records_the_leakage_audit(built):
    sc = built["scorer"]
    pred = sc.predict(built["records"])
    fams = sorted(f for f in built["records"]["attack_family"].unique() if f != LABEL_BENIGN)
    cfg = built["cfg"]

    def per_family(fam):
        """Same shape as run_experiment.run_leave_one_family_out's closure: the family is removed
        from every permitted stage, memory+threshold are rebuilt, then it is scored as unseen."""
        fold_records = built["records"][built["records"]["attack_family"] != fam]
        s2 = Member3Scorer.build(cfg, built["adapter"], built["adapter_meta"], fold_records)
        p2 = s2.verify_alerts(s2.predict(built["records"]))
        row = ev.per_attack_family(p2, cfg)
        row = row[row["attack_family"] == fam].iloc[0]
        return {"f1": float(row["f1"]), "precision": float(row["precision"]), "recall": float(row["recall"]),
                "threshold": float(s2.threshold.value),
                "memory_changed": int(not np.array_equal(s2.memory.vectors().astype("float32"),
                                                         sc.memory.vectors().astype("float32"))),
                "n_samples": int(row["n_positive"])}

    df, audit = ev.leave_one_family_out(built["records"], {**cfg, "_baseline_threshold": float(sc.threshold.value)},
                                       fams, per_family)
    assert len(df) == len(fams) and set(df.columns) >= {"held_out_family", "rows_removed_from_permitted_stages",
                                                        "rows_total_for_family", "status", "tau_delta_vs_baseline"}
    assert (df["status"] == "ok").all()
    assert np.allclose(df["tau_delta_vs_baseline"].to_numpy(dtype=float), 0.0, atol=1e-12)
    assert audit["rows_removed_from_permitted_stages"] == 0 and audit["n_folds"] == len(fams)
    assert int(df["rows_total_for_family"].min()) > 0        # the family really was held out of fitting
    assert audit["leakage_detected"] is False
    assert audit["max_abs_tau_delta"] == pytest.approx(0.0) and audit["memory_changed_folds"] == 0
    assert int(df["rows_removed_from_permitted_stages"].sum()) == 0       # attacks never entered memory/calibration
    assert set(df["held_out_family"]) == set(fams)


def test_bootstrap_ci_brackets_the_mean_and_reports_n():
    rng = np.random.default_rng(4)
    v = rng.normal(0.5, 0.1, 400)
    ci = bootstrap_ci(v, n_boot=200, seed=0)
    assert set(ci) == {"point", "low", "high", "n", "n_boot"}
    assert ci["n"] == 400 and ci["n_boot"] == 200 and ci["low"] < ci["point"] < ci["high"]
    assert ci["low"] <= v.mean() <= ci["high"] and abs(ci["point"] - v.mean()) < 1e-9
    assert ci["high"] - ci["low"] < 0.05
    one = bootstrap_ci(np.array([0.5] * 20), n_boot=50, seed=0)
    assert one["point"] == 0.5 and one["low"] == one["high"] == 0.5      # degenerate but honest
    assert bootstrap_ci(np.array([]), n_boot=10)["n"] == 0               # empty -> nan/n, no crash


# --------------------------------------------------------------------------- #
# tables, figures, manifest
# --------------------------------------------------------------------------- #
def test_tidy_tables_carry_dataset_split_config_and_counts():
    rec = {"precision": 0.8, "recall": 0.6, "n_samples": 100, "notes": ""}
    df = pd.DataFrame(ev.tidy("T1_overall", rec, dataset="DS", split="test", config_id="cfg-1",
                              data_source="DS-TEST", n_samples=100))
    assert set(df.columns) >= {"table", "metric", "value", "dataset", "split", "config_id", "n_samples",
                               "data_source", "row_key"}
    assert set(df["metric"]) >= {"precision", "recall", "n_samples"} and len(df) == 4
    keyed = pd.DataFrame(ev.tidy("T1_overall", rec, dataset="DS", split="test", config_id="c",
                                data_source="d", n_samples=1, row_key="r1"))
    assert set(keyed["row_key"]) == {"r1"}
    notrun = pd.DataFrame(ev.tidy("T1_overall", {"precision": ev.NOT_RUN, "n_samples": 0}, dataset="DS",
                                  split="test", config_id="c", data_source="d", n_samples=0))
    assert notrun["value"].iloc[0] == ev.NOT_RUN


def test_build_paper_tables_covers_nine_tables_and_blocked_ones_stay_visible():
    ids = ["TABLE1_overall_performance", "TABLE2_cross_dataset", "TABLE3_attack_family", "TABLE4_zero_day",
           "TABLE5_threshold_ablation", "TABLE6_debounce", "TABLE7_memory_adaptation", "TABLE8_fidelity_plus",
           "TABLE9_latency"]
    ctx = {"dataset": "DS", "split": SPLIT_TEST, "config_id": "c1", "data_source": "DS-TEST", "n_samples": 10}
    bundles = [(t, f"caption {t}", pd.DataFrame({"f1": [0.5], "n_samples": [10]}), ctx) for t in ids[:8]]
    bundles.append((ids[8], "blocked", pd.DataFrame(), {"reason": "no latency benchmark run", **ctx}))
    df, md = ev.build_paper_tables(bundles)
    assert set(df["table"]) == set(ids)
    assert md.count("## ") == 9 and all(f"## {t}" in md for t in ids)
    blocked = df[df["table"] == ids[8]]
    assert blocked.loc[blocked["metric"] == "result", "value"].tolist() == [ev.NOT_RUN]
    assert "no latency benchmark" in " ".join(blocked["value"].astype(str))
    assert (df["dataset"] == "DS").all() and (df["config_id"] == "c1").all() and (df["n_samples"] >= 0).all()


def test_figures_render_and_blocked_ones_are_recorded(tmp_path):
    status = {}
    rec = pd.DataFrame({"label": [LABEL_ATTACK, LABEL_ATTACK, LABEL_BENIGN, LABEL_BENIGN],
                        "anomaly_score": [0.9, 0.8, 0.02, 0.01], "threshold": [0.1] * 4,
                        "attack_family": ["Exploits", "Generic", "benign", "benign"],
                        "confirmed_alert": [True, False, False, False], "alert_raised": [True, False, False, False],
                        "detection_delay_steps": [2, np.nan, np.nan, np.nan],
                        "confidence": [0.99, 0.9, 0.1, 0.05], "split": ["test"] * 4,
                        "is_anomaly": [True, True, False, False], "memory_version": [0] * 4,
                        "threshold_method": ["confidence_aware"] * 4, "dataset_id": ["DS-TEST"] * 4,
                        "nearest_normal_similarity": [0.1, 0.2, 0.98, 0.99]})
    ev.figure_roc_pr(rec, str(tmp_path), status)
    ev.figure_score_distribution(rec, str(tmp_path), status)
    ev.figure_family_breakdown(ev.per_attack_family(rec, raw_cfg()), str(tmp_path), status)
    assert any(f.endswith(".png") for f in status) and tmp_path.exists()
    files = sorted(os.listdir(tmp_path))
    assert all(f.endswith(".png") for f in files)
    for f in files:
        assert os.path.getsize(os.path.join(tmp_path, f)) > 10_000       # real plots, not stubs
    missing = {}
    ev.figure_cross_dataset(pd.DataFrame(), str(tmp_path), missing)      # one dataset only -> NOT RUN
    assert missing and all(str(v).startswith(ev.NOT_RUN) for v in missing.values())
    assert "NF-ToN-IoT-v3" in list(missing.values())[0]                 # says exactly what unblocks it


def test_manifest_records_provenance_and_failed_stages(tmp_path):
    path = str(tmp_path / "manifest.json")
    ev.write_manifest(path, {"stages": {"evaluation": {"ok": False, "error": "ContractError: missing adapter"}},
                             "data_source": "DS-TEST"})
    with open(path) as fh:
        m = json.load(fh)
    assert m["stages"]["evaluation"]["ok"] is False
    assert "missing adapter" in m["stages"]["evaluation"]["error"]      # failures recorded, not hidden


# --------------------------------------------------------------------------- #
# contract enforcement
# --------------------------------------------------------------------------- #
def test_records_roundtrip_and_loud_failures(tmp_path):
    rec, _ = make_records(n_train=12, n_cal=10, families=("Exploits",))
    p = str(tmp_path / "records.npz")
    save_records(rec, p)
    back = load_records(p)
    assert set(back.columns) == set(rec.columns) and len(back) == len(rec)
    assert np.allclose(np.vstack(list(back["embedding"])), np.vstack(list(rec["embedding"])), atol=1e-5)
    assert back["label"].tolist() == rec["label"].tolist()
    with pytest.raises(ContractError, match="missing required columns"):
        validate_records(back.drop(columns=["split"]))
    with pytest.raises(ContractError, match="dimension mismatch"):
        validate_records(back, expected_dim=DIM + 1)
    ragged = back.copy()
    ragged["embedding"] = list(ragged["embedding"])
    ragged["embedding"] = [np.zeros(DIM - 1)] + list(ragged["embedding"].iloc[1:])
    with pytest.raises(ContractError, match="ragged"):
        validate_records(ragged)
    with pytest.raises(ContractError, match="label column"):
        bad = back.copy()
        bad.loc[0, "label"] = "maybe"
        validate_records(bad)
    csv_path = str(tmp_path / "records.csv")
    save_records(back, csv_path)
    assert len(load_records(csv_path)) == len(back)


def test_graph_file_formats_match_the_contract(tmp_path):
    rec, graphs = make_records(n_train=3, n_cal=3, families=("Exploits",))
    sid = sorted(graphs)[0]
    for name in ("graphs.json", "graphs.npz"):
        p = str(tmp_path / name)
        save_graphs({sid: graphs[sid]}, p)
        loaded = load_graphs(p)
        assert list(loaded) == [sid]
        assert len(loaded[sid]["snapshots"]) == N_SNAP
        assert loaded[sid]["snapshots"][0]["node_ids"] == graphs[sid]["snapshots"][0]["node_ids"]
    with pytest.raises(ContractError, match="unsupported graph file"):
        load_graphs(str(tmp_path / "graphs.h5"))


def test_contamination_guards_fire():
    rec, _ = make_records(n_train=10, n_cal=10, families=("Exploits",))
    dirty = rec.copy()
    dirty.loc[dirty["split"] == SPLIT_MEMORY, "label"] = LABEL_ATTACK
    with pytest.raises(ContractError, match="CONTAMINATION"):
        assert_splits_clean(dirty)
    leaked = rec.copy()
    test_ids = leaked.loc[leaked["split"] == SPLIT_TEST, "sample_id"].head(2).tolist()
    leaked.loc[leaked["split"] == SPLIT_MEMORY, "sample_id"] = test_ids[0]
    with pytest.raises(ContractError, match="test sample_ids"):
        assert_splits_clean(leaked)
    wrong_ds = rec.copy()
    wrong_ds.loc[wrong_ds["split"] == SPLIT_MEMORY, "dataset_id"] = "DS-TEST"
    with pytest.raises(ContractError, match="not permitted calibration sources"):
        assert_splits_clean(wrong_ds, permitted_reference_datasets=["DS-TRAIN"])


def test_adapter_loader_requires_the_upstream_file(tmp_path):
    cfg = raw_cfg()
    with pytest.raises(ContractError, match="Member-2 adapter not found"):
        from member3.contract import load_adapter
        load_adapter(cfg)
    art = tmp_path / "member2_artifacts"
    art.mkdir()
    (art / "model_adapter.py").write_text(ADAPTER_SOURCE)
    from member3.contract import load_adapter
    adapter, meta = load_adapter(cfg, artifacts_dir=str(art))
    assert meta["embedding_dim"] == DIM and meta["adapter_origin"] == "member2_adapter"
    assert adapter.metadata()["embedding_dim"] == DIM
    with pytest.raises(ContractError, match="dimension mismatch"):
        validate_adapter(TinyAdapter(dim=8), expected_dim=DIM)
    (art / "model_adapter.py").write_text("x = 1\n")
    with pytest.raises(ContractError, match="expected class"):
        load_adapter(cfg, artifacts_dir=str(art))


def test_stream_derivation_when_upstream_omits_it():
    rec, _ = make_records(n_train=5, n_cal=5, families=("Exploits",))
    bare = rec.drop(columns=["stream_id", "stream_pos", "episode_id"])
    out = assign_streams(bare, window_seconds=30.0, seed=0)
    assert {"stream_id", "stream_pos", "episode_id"} <= set(out.columns)
    for _, g in out.groupby("stream_id"):
        pos = sorted(int(v) for v in g["stream_pos"])
        assert len(pos) == len(set(pos)) and np.all(np.diff(pos) == 1)      # one dense, ordered run per stream
    att = out[(out["split"] == SPLIT_TEST) & (out["label"] == LABEL_ATTACK)].sort_values("stream_pos")
    ids = set(att["episode_id"]) - {"benign"}
    assert ids and all(i.split("#")[0] in {"Exploits", "Reconnaissance"} for i in ids)
    for i, g in att.groupby("episode_id"):
        assert g["stream_pos"].diff().dropna().eq(1).all()        # each burst is one contiguous run
    assert len(att) == 5 and len(ids) == 1 and "no-such-window" not in set(att["sample_id"])


# --------------------------------------------------------------------------- #
# latency + API
# --------------------------------------------------------------------------- #
def test_latency_measurement_reports_stages_and_conditions(built):
    sc, rec = built["scorer"], built["records"]
    X = np.vstack(list(rec[rec["split"] == SPLIT_TEST]["embedding"]))
    from member3.inference import benchmark_latency
    out = benchmark_latency(sc, X, n_samples=10, n_repeat=2, warmup=1)
    for key in ("B_faiss_retrieval", "C_thresholding", "D_alert_verification", "D_alert_verification_batch",
                "E_member3_total"):
        assert key in out["per_sample_ms"], sorted(out["per_sample_ms"])
        assert set(out["per_sample_ms"][key]) >= {"mean", "median", "p95", "p99", "n"}
        assert out["per_sample_ms"][key]["mean"] >= 0.0 and out["per_sample_ms"][key]["n"] >= 10
        assert out["per_sample_ms"][key]["max"] >= out["per_sample_ms"][key]["mean"] > 0.0
    assert out["stages_not_measured"] == ["A_embedding"] and out["A_note"].startswith(ev.NOT_RUN)
    assert out["conditions"]["k"] == built["cfg"]["retrieval"]["k"]
    assert out["conditions"]["memory_size"] == sc.memory.stats.n
    assert "device" in out["conditions"] and "cpu_count" in out["conditions"]
    assert out["conditions"]["throughput_samples_per_s"] > 0
    tbl = ev.latency_table(out, config_id="c1", data_source="DS-TEST")
    assert set(tbl.columns) >= {"stage", "stage_description", "mean_ms", "p95_ms", "k", "index_type",
                               "memory_size", "cpu_count", "throughput_samples_per_s", "config_id", "data_source"}
    assert set(tbl["stage"]) == set(out["per_sample_ms"]) and "A_embedding" not in set(tbl["stage"])
    assert out["A_note"].startswith(ev.NOT_RUN)          # encode not measured: cached embeddings in, graphs out
    test_rows = rec[rec["split"] == SPLIT_TEST]
    graphs_aligned = [built["graphs"][s] for s in test_rows["sample_id"]]
    out2 = benchmark_latency(sc, np.vstack(list(test_rows["embedding"])), graphs=graphs_aligned,
                             n_samples=8, n_repeat=1, warmup=1)
    assert "A_embedding" not in out2["stages_not_measured"]
    assert out2["per_sample_ms"]["A_embedding"]["n"] == 8
    tbl2 = ev.latency_table(out2, config_id="c1", data_source="DS-TEST")
    misaligned = benchmark_latency(sc, X, graphs=graphs_aligned[:3], n_samples=4, n_repeat=1, warmup=1)
    assert misaligned["stages_not_measured"] == ["A_embedding"]
    assert "same order" in misaligned["A_note"]      # misalignment is reported, not crashed on
    assert "A_embedding" in set(tbl2["stage"])          # with graphs supplied, encode time is measurable
    assert tbl2.loc[tbl2["stage"] == "A_embedding", "mean_ms"].notna().all()


def test_api_serves_score_explain_and_refuses_bad_dim(built):
    pytest.importorskip("fastapi")
    import yaml
    from fastapi.testclient import TestClient
    from member3.api import create_app

    cfg_path = os.path.join(built["dir"], "config.yaml")
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(built["cfg"], fh)
    client = TestClient(create_app(cfg_path))

    h = client.get("/health").json()
    assert h["status"] == "ok" and h["memory_size"] == built["scorer"].memory.stats.n
    assert h["threshold"] == pytest.approx(float(built["scorer"].threshold.value))
    assert h["index_type"] == "numpy" and h["immutable_memory"] is True
    assert h["dataset_label"] == "DS-TEST" and h["sample_ids"] and h["threshold_method"] == "confidence_aware"

    sid = built["records"].query("split == 'test' and label == 'attack'")["sample_id"].iloc[4]
    r = client.post("/score", json={"sample_id": str(sid)}).json()
    assert r["anomaly_score"] > r["threshold"] and 0.0 <= r["confidence"] <= 1.0
    assert r["alert"]["alert_state"] in ("candidate", "confirmed") and len(r["neighbours"]) == 4
    assert r["scoring_definition"] == "1 - max_j cos(z, m_j)" and r["model_weights_updated"] is False
    assert r["memory_update"] == "disabled" and r["dataset_label"] == "DS-TEST"
    assert all(0.0 <= n["similarity"] <= 1.0 for n in r["neighbours"])

    # stream state advances: candidate 2/3 -> confirmed 3/3
    sids = built["records"].query("split == 'test' and label == 'attack'")["sample_id"].tolist()[:3]
    states = [client.post("/score", json={"sample_id": s, "stream_id": "api-stream"}).json()["alert"] for s in sids]
    assert [x["consecutive_anomalies"] for x in states] == [1, 2, 3]
    assert states[2]["alert_state"] == "confirmed" and states[2]["n_confirmed_alerts"] == 1

    e = client.post("/explain", json={"sample_id": str(sid), "stream_id": "api-stream2"}).json()["explanation"]
    assert e["top_nodes"] and e["top_features"] and e["summary"] and e["top_edges"]
    assert e["anomaly_score"] == pytest.approx(r["anomaly_score"], abs=1e-3)

    bad = client.post("/score", json={"embedding": [0.1] * (DIM + 1)})
    assert bad.status_code == 400 and "dimension" in bad.text.lower()
    empty = client.post("/score", json={})
    assert empty.status_code == 400 and "sample_id" in empty.text.lower()
    unknown = client.post("/score", json={"sample_id": "no-such-window"})
    assert unknown.status_code == 400 and "sample_id" in unknown.text.lower()
    # /metrics serves the offline results file: absent -> 404 with the fix, present -> verbatim payload
    assert client.get("/metrics").status_code == 404
    assert "run_experiment" in client.get("/metrics").json()["detail"]
    rdir = os.path.join(built["dir"], "results")
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "metrics.json"), "w") as fh:
        json.dump({"overall_sample_level": {"f1": 0.5}}, fh)
    built["cfg"]["output"]["results_dir"] = rdir
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(built["cfg"], fh)
    client2 = TestClient(create_app(cfg_path))
    assert client2.get("/metrics").json() == {"overall_sample_level": {"f1": 0.5}}
    page = client.get("/").text
    assert "<html" in page.lower() and "Member 3" in page
    assert "memory" in page and "threshold" in page.lower()
    assert "cdn." not in page and "http://" not in page.replace("http://localhost", "")   # self-contained view


def test_api_memory_update_is_off_by_default(built):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from member3.api import create_app
    import yaml
    cfg_path = os.path.join(built["dir"], "config.yaml")
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(built["cfg"], fh)
    client = TestClient(create_app(cfg_path))
    st0 = built["scorer"].memory.stats
    n0, v0 = st0.n, st0.version
    sid = built["records"].query("split == 'test' and label == 'benign'")["sample_id"].iloc[0]
    body = client.post("/score", json={"sample_id": str(sid), "allow_memory_update": True}).json()
    assert body["memory_update"] == "disabled" and body["model_weights_updated"] is False
    assert built["scorer"].memory.stats.n == n0 and built["scorer"].memory.stats.version == v0
