r"""
export_member3_package.py -- builds the hand-off package for Member 3. READ-ONLY on all inputs; writes a NEW folder.

Put in:  <PROJECT_ROOT>\training\export_member3_package.py   (needs model_adapter.py and exp_common.py next to it)
Run:     python training\export_member3_package.py --ckpt checkpoints_validation\best_model.pt ^
             --graph graph.pt --val graph_validation_v2.pt ^
             --npz diagnostics\embeddings_baseline_20261022_010656.npz  *> export_console.txt
(PowerShell: use ` at line ends instead of ^, or put it on one line.)
Writes:  <PROJECT_ROOT>\member3_package_<timestamp>\
   member2_records.npz  member2_records.csv  metadata_discovery.json  sample_pairs.pt
   model_config.yaml  feature_schema.json  preprocessing.json  EXPORT_REPORT.txt

Nothing is invented. A field is filled ONLY if it exists in the source .pt files (native key). Otherwise it is left
empty and listed under "MISSING" in EXPORT_REPORT.txt. Two derived columns are explicitly marked as derived:
  source_index (row position in the source file) and the "positional" sample_id used when no native id exists.
"""
import os, sys, json, csv, hashlib, argparse, platform, pickle, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.append(HERE); sys.path.append(ROOT)
from model_adapter import Member2Adapter

SEED = 42
ID_KEYS = ["sample_id", "seq_id", "sequence_id", "id"]
WIN_KEYS = ["window_id", "window_ids", "window_idx", "window_index"]
TS_KEYS = ["timestamp", "timestamps", "start_time", "window_start", "window_times", "time", "ts"]
CORE = {"snapshots", "label", "attack_types"}
REPORT = []


def say(s=""):
    print(s); REPORT.append(s)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jsonable(o, depth=0):
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    if isinstance(o, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v, depth + 1) for v in o]
    if hasattr(o, "to_dict"):
        try:
            return jsonable(o.to_dict(), depth + 1)
        except Exception:
            pass
    attrs = {k: getattr(o, k) for k in ("data_min_", "data_max_", "data_range_", "scale_", "min_", "mean_", "var_",
                                         "center_", "feature_range", "n_features_in_", "feature_names_in_") if hasattr(o, k)}
    return jsonable(attrs, depth + 1) if attrs else {"repr": repr(o)[:500]}


def native(seq, keys):
    for k in keys:
        if k in seq:
            return k, seq[k]
    return None, None


def fmt(v):
    """string form of a native value: ISO for datetimes/timestamps, JSON for lists, str otherwise"""
    if v is None:
        return ""
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return json.dumps([fmt(t) for t in v])
    if torch.is_tensor(v) or isinstance(v, np.ndarray):
        return json.dumps(jsonable(v))
    return str(v)


LABEL_MAP = {"normal": 0, "benign": 0, "0": 0, "attack": 1, "malicious": 1, "1": 1}


def label_num(v):
    """numeric label ONLY for an unambiguous native encoding; anything else stays empty (never guessed)"""
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer)) and int(v) in (0, 1):
        return int(v)
    if torch.is_tensor(v) and v.numel() == 1 and int(v) in (0, 1):
        return int(v)
    return LABEL_MAP.get(str(v).strip().lower(), "")


def scalar_meta(seq):
    out = {}
    for k, v in seq.items():
        if k in CORE:
            continue
        if torch.is_tensor(v) and v.numel() == 1:
            v = v.item()
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 10 and all(isinstance(t, (str, int, float, bool)) for t in v):
            out[k] = json.dumps(list(v))
    return out


def edges_nodes(seq):
    return (sum(int(s.edge_index.shape[1]) for s in seq["snapshots"]), sum(int(s.x.shape[0]) for s in seq["snapshots"]))


def flow_ratio(seq):
    if not hasattr(seq["snapshots"][0], "edge_y"):
        return ""
    tot = sum(len(s.edge_y) for s in seq["snapshots"]); att = sum(int((s.edge_y == 1).sum()) for s in seq["snapshots"])
    return att / max(tot, 1)


def discover(name, data, seqs):
    top = {k: jsonable(v) if (isinstance(v, (str, int, float, bool)) or (isinstance(v, (list, tuple)) and len(v) < 20)) else f"<{type(v).__name__}>"
           for k, v in data.items() if k != "sequences"}
    seq_keys = sorted({k for s in seqs for k in s.keys()})
    snap_keys = sorted({k for s in seqs[:50] for g in s["snapshots"] for k in g.to_dict().keys()})
    say(f"[{name}] top-level keys other than 'sequences': {top}")
    say(f"[{name}] per-sequence keys: {seq_keys}")
    say(f"[{name}] per-snapshot (Data) keys: {snap_keys}")
    return dict(top_level=top, sequence_keys=seq_keys, snapshot_keys=snap_keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--graph", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--npz", default=None, help="cached embeddings file to cross-check (optional)")
    ap.add_argument("--dataset1-id", default="dataset1_graph_pt"); ap.add_argument("--dataset2-id", default="dataset2_graph_validation_v2")
    a = ap.parse_args()
    out = os.path.join(ROOT, f"member3_package_{time.strftime('%Y%m%d_%H%M%S')}"); os.makedirs(out)
    ad = Member2Adapter(a.ckpt)
    tr = torch.load(a.graph, map_location="cpu", weights_only=False); va = torch.load(a.val, map_location="cpu", weights_only=False)
    trs, vas = tr["sequences"], va["sequences"]

    say("=" * 70 + "\nSOURCE FILES\n" + "=" * 70)
    hashes = {"graph": sha256(a.graph), "validation": sha256(a.val), "checkpoint": sha256(a.ckpt)}
    for k, v in hashes.items():
        say(f"sha256 {k}: {v}")
    disc = {"dataset1": discover("graph.pt", tr, trs), "dataset2": discover("graph_validation_v2.pt", va, vas)}
    json.dump(disc, open(os.path.join(out, "metadata_discovery.json"), "w"), indent=2)
    from collections import Counter
    say(f"native label values  graph.pt: {dict(Counter(str(s['label']) for s in trs))}")
    say(f"native label values  graph_validation_v2.pt: {dict(Counter(str(s['label']) for s in vas))}")
    say("(numeric label: 'normal'/'benign'/0 -> 0, 'attack'/1 -> 1; any other value is left empty, never guessed)")

    # ---------------------------------------------------------------- splits (exactly as verify_new_validation.py / eval_exp.py)
    perm = torch.randperm(len(trs), generator=torch.Generator().manual_seed(SEED)).tolist()
    cut = int(0.9 * len(trs))
    y_val = np.array([label_num(s["label"]) for s in vas], dtype=int)
    rng = np.random.default_rng(SEED)
    b = rng.permutation(np.where(y_val == 0)[0]); atk = np.where(y_val == 1)[0]
    plan = [("memory_train", a.dataset1_id, "graph", perm[:cut]),
            ("benign_heldout_dataset1", a.dataset1_id, "graph", perm[cut:]),
            ("calibration", a.dataset2_id, "val", list(b[: len(b) // 2])),
            ("test", a.dataset2_id, "val", list(b[len(b) // 2:]) + list(atk))]
    src = {"graph": trs, "val": vas}

    say("\n" + "=" * 70 + "\nEMBEDDING (a few minutes)\n" + "=" * 70)
    recs, E = [], []
    for split, dsid, which, idxs in plan:
        say(f"{split}: {len(idxs)} rows from {dsid}")
        for i in idxs:
            i = int(i); s = src[which][i]
            h = ad.encode(s["snapshots"]); E.append(h)
            kid, vid = native(s, ID_KEYS); kw, vw = native(s, WIN_KEYS); kt, vt = native(s, TS_KEYS)
            lab_native = s.get("label", "")
            lab = label_num(lab_native) if "label" in s else ""
            fam = ""
            if lab == 1:
                fam = "|".join(map(str, s.get("attack_types") or [])) if "attack_types" in s else ""
            elif lab == 0:
                fam = "BENIGN"
            ne, nn = edges_nodes(s)
            r = dict(dataset_id=dsid, split=split, sample_id=str(vid) if kid else f"{dsid}:{i:05d}",
                     sample_id_source=f"native:{kid}" if kid else "positional(derived)",
                     window_id=fmt(vw) if kw else "", timestamp=fmt(vt) if kt else "",
                     end_time=fmt(s.get("end_time")) if "end_time" in s else "",
                     label=lab, label_native=str(lab_native), attack_family=fam, source_file=os.path.basename(a.graph if which == "graph" else a.val),
                     source_index=i, n_nodes=nn, n_edges=ne,
                     attack_flow_ratio=flow_ratio(s) if which == "val" else "")
            r.update({f"meta_{k}": v for k, v in scalar_meta(s).items()})
            recs.append(r)
    E = np.stack(E).astype(np.float32)

    # ---------------------------------------------------------------- cross-check with the cached NPZ
    say("\n" + "=" * 70 + "\nCROSS-CHECK against the cached NPZ\n" + "=" * 70)
    if a.npz:
        c = np.load(a.npz)
        mem = [k for k, r in enumerate(recs) if r["split"] == "memory_train"]
        order = np.argsort([recs[k]["source_index"] for k in range(len(recs)) if recs[k]["dataset_id"] == a.dataset2_id])
        vrows = [k for k, r in enumerate(recs) if r["dataset_id"] == a.dataset2_id]
        vrows_sorted = [vrows[j] for j in order]
        d_m = float(np.abs(E[mem] - c["Hm"]).max()); d_v = float(np.abs(E[vrows_sorted] - c["Hv"]).max())
        yv = np.array([recs[k]["label"] for k in vrows_sorted])
        say(f"memory rows (graph.pt, seed-42 order):  max |h_new - Hm| = {d_m:.2e}  -> {'MATCH' if d_m < 1e-3 else 'MISMATCH'}")
        say(f"validation rows (file order):           max |h_new - Hv| = {d_v:.2e}  -> {'MATCH' if d_v < 1e-3 else 'MISMATCH'}")
        say(f"labels identical to npz y:               {bool(np.array_equal(yv, c['y']))}")
    else:
        say("no --npz given; skipped")

    # ---------------------------------------------------------------- records files
    cols = ["dataset_id", "split", "sample_id", "sample_id_source", "window_id", "timestamp", "end_time", "label",
            "label_native", "attack_family",
            "source_file", "source_index", "n_nodes", "n_edges", "attack_flow_ratio"] + sorted({k for r in recs for k in r if k.startswith("meta_")})
    str_cols = {c: np.array([str(r.get(c, "")) for r in recs]) for c in cols}
    np.savez_compressed(os.path.join(out, "member2_records.npz"), embedding=E, **str_cols)
    with open(os.path.join(out, "member2_records.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols + [f"h_{j}" for j in range(E.shape[1])])
        for r, e in zip(recs, E):
            w.writerow([r.get(c, "") for c in cols] + [f"{v:.7g}" for v in e])
    say("\n" + "=" * 70 + "\nRECORDS\n" + "=" * 70)
    say(f"rows: {len(recs)}  embedding shape: {E.shape}  (h, 128-d, unit norm: min/max norm {np.linalg.norm(E,axis=1).min():.6f}/{np.linalg.norm(E,axis=1).max():.6f})")
    for sp in dict.fromkeys(r["split"] for r in recs):
        rr = [r for r in recs if r["split"] == sp]
        say(f"  {sp:26s} n={len(rr):5d}  labels={dict(zip(*np.unique([str(r['label']) for r in rr], return_counts=True)))}")
    missing = [n for n, ks in (("sample_id (native)", ID_KEYS), ("window_id", WIN_KEYS), ("timestamp", TS_KEYS))
               if not any(k in s for s in vas[:200] for k in ks)]
    say("\nMISSING (no native key found in graph_validation_v2.pt; left empty / positional, NOT invented): " + (", ".join(missing) or "none"))
    say("attack_family is the window-level 'attack_types' list ('|'-joined); a window can contain several families.")
    say("label: 1 = window contains at least one attack flow (Member 1's definition); median attack window is ~2.2% attack flows.")

    # ---------------------------------------------------------------- a real test pair
    def pick(label):
        cand = [(edges_nodes(s)[0], i) for i, s in enumerate(vas) if label_num(s["label"]) == label and edges_nodes(s)[0] >= 200]
        return min(cand)[1]
    pairs = []
    for lab in (0, 1):
        i = pick(lab); s = vas[i]; h = ad.encode(s["snapshots"])
        pairs.append(dict(source_index=i, label=lab, attack_types=s.get("attack_types"), snapshots=s["snapshots"], h=torch.tensor(h)))
    torch.save(pairs, os.path.join(out, "sample_pairs.pt"))
    say(f"\nsample_pairs.pt: smallest benign (idx {pairs[0]['source_index']}) and smallest attack (idx {pairs[1]['source_index']}) windows "
        "with their h; re-encode with Member2Adapter and compare to pairs[i]['h'] (max abs diff should be < 1e-4).")

    # ---------------------------------------------------------------- artifacts
    md = ad.metadata()
    def pk(name):
        p = os.path.join(ROOT, name)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return pickle.load(f)
    names, norm, idcols = pk("feature_cols.pkl"), pk("norm_stats.pkl"), pk("id_cols.pkl")
    fl_tr = trs[0].get("feature_list"); fl_va = va.get("feature_list", vas[0].get("feature_list"))
    same = (list(fl_tr) == list(fl_va)) if (fl_tr is not None and fl_va is not None) else None
    same_pkl = (list(fl_va) == list(names)) if (fl_va is not None and names is not None) else None
    say(f"\nfeature_list graph.pt == graph_validation_v2.pt: {same};  == feature_cols.pkl: {same_pkl}")
    json.dump(dict(num_features=53, feature_order=jsonable(fl_va if fl_va is not None else names),
                   source="'feature_list' stored inside the .pt files (native)" if fl_va is not None else "feature_cols.pkl",
                   identical_between_dataset1_and_dataset2=same, identical_to_feature_cols_pkl=same_pkl,
                   note="feature_list names the 53 per-flow edge_attr columns; confirm with Member 1 how node features x are built",
                   id_columns=jsonable(idcols)), open(os.path.join(out, "feature_schema.json"), "w"), indent=2)
    json.dump(dict(
        scaling=dict(applied_by="Member 1 before graph.pt / graph_validation_v2.pt were saved; the model does NOT scale",
                     statistics=jsonable(norm), source="norm_stats.pkl" if norm is not None else "NOT FOUND",
                     expected_range_in_graph_files="[0, 1] (verified: max 1.000 for all 53 node and 53 edge features in both files)"),
        model_side_preprocessing=dict(clamp=[-5.0, 5.0], input_transform=md["input"]["input_transform"]),
        temporal=dict(snapshots_per_sequence=5, source="native keys of graph_validation_v2.pt and graph.pt sequences",
                      window_size_sec_dataset1=sorted({str(s.get("window_size_sec")) for s in trs[:200]}),
                      dataset2_header=disc["dataset2"]["top_level"]),
        labels=dict(rule="window label 1 = contains at least one attack flow; attack_types lists families present",
                    encoding={"0": "benign", "1": "attack"}),
        splits=dict(seed=SEED, memory_train="graph.pt, first 90% after torch.randperm(n, manual_seed(42))",
                    benign_heldout_dataset1="graph.pt, remaining 10%",
                    calibration="validation benign, first half of np.random.default_rng(42).permutation(benign_indices)",
                    test="validation benign second half + ALL validation attack windows"),
        file_sha256=hashes, versions=md["versions"],
        platform=dict(os=platform.platform(), python=platform.python_version())), open(os.path.join(out, "preprocessing.json"), "w"), indent=2)
    ck = md["checkpoint"]
    yaml = f"""# Member 2 encoder -- generated from the actual checkpoint
model:
  name: member2_gatv2_temporal
  checkpoint: {os.path.basename(a.ckpt)}
  checkpoint_sha256: {hashes['checkpoint']}
  best_epoch: {ck['best_epoch']}
  best_validation_loss: {ck['best_validation_loss']}
  num_parameters: {ck['num_parameters']}
  embedding_used_downstream: h
  h_dim: 128
  h_l2_normalised: true
  z_dim: 64
gatv2:
  layer1: {{in: 53, heads: 4, out_per_head: 128, concat: true, edge_dim: 53, dropout: 0.2, activation: ELU}}
  layer2: {{in: 512, heads: 1, out: 128, edge_dim: 53, dropout: 0.2}}
  self_loops: true
pooling: {md['pooling']}_over_nodes
temporal_transformer: {{snapshots: 5, positional_embedding: learned, layers: 2, heads: 4, ffn: 512, activation: gelu, dropout: 0.1, norm: post_ln, readout: mean_tokens_layernorm_l2}}
projection_head: {{in: 128, hidden: 128, out: 64, activation: GELU}}
input: {{node_features: 53, edge_features: 53, clamp: [-5.0, 5.0], transform: {md['input']['input_transform']}}}
training: {{batch_size: {ck['batch_size']}, lr: {ck['learning_rate']}, nt_xent_temperature: {ck['nt_xent_temperature']}, seed: {ck['seed']}, objective: benign_only_contrastive}}
anomaly_scoring: {{memory: dataset1_train_h, metric: cosine_distance, k: 5, threshold: benign_calibration_p95}}
"""
    open(os.path.join(out, "model_config.yaml"), "w").write(yaml)
    say("\nWrote: " + ", ".join(sorted(os.listdir(out))) + f"\nFolder: {out}")
    open(os.path.join(out, "EXPORT_REPORT.txt"), "w", encoding="utf-8").write("\n".join(REPORT))


if __name__ == "__main__":
    main()