r"""
diagnose_attack_signal.py -- READ-ONLY. Why do attacks look "more normal" than benign on the corrected file?

Put in:  <PROJECT_ROOT>\training\diagnose_attack_signal.py
Run:     python training\diagnose_attack_signal.py --val "D:\DL_ZeroDay_Member2\graph_validation_v2.pt" *> signal_console.txt
Writes:  <PROJECT_ROOT>\diagnostics\signal_<timestamp>.json and embeddings_<timestamp>.npz (new files each run)

Checks
  1. AUC restricted to attack windows with a minimum share of attack flows (dilution test)
  2. AUC within fine edge-count bins (size control)
  3. Diagnostic linear probes with 5-fold CV on h, z, raw features, size  (uses labels; NOT a detector)
  4. kNN on plain averaged input features (is the encoder discarding signal?)
Caveat: probe CV folds are random; if consecutive windows overlap in time the probe AUC is optimistic.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from models.gatv2_encoder import GATv2Encoder
from models.temporal_transformer import TemporalTransformer
from models.projection_head import ProjectionHead
from models.global_pooling import global_mean_pooling

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, K = 42, 5
CKPT = os.path.join(ROOT, "checkpoints_validation", "best_model.pt")
OUT = {}


def sec(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def load_models():
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    gat = GATv2Encoder(53, 128, 4, 0.2).to(DEV); tr = TemporalTransformer(128, 4, 2, 0.1).to(DEV)
    pr = ProjectionHead(128, 128, 64).to(DEV)
    gat.load_state_dict(ck["gat_state_dict"]); tr.load_state_dict(ck["transformer_state_dict"])
    pr.load_state_dict(ck["projection_state_dict"])
    for m in (gat, tr, pr):
        m.eval()
    return gat, tr, pr


@torch.no_grad()
def embed(models, seqs):
    gat, tr, pr = models
    H, Z = [], []
    for s in seqs:
        toks = []
        for g in s["snapshots"]:
            x = g.x.to(DEV).clamp(-5, 5); ea = g.edge_attr.to(DEV).clamp(-5, 5)
            toks.append(global_mean_pooling(gat(x, g.edge_index.to(DEV), ea)))
        h = tr(torch.cat(toks, 0).unsqueeze(0))
        H.append(h.squeeze(0).cpu()); Z.append(pr(h).squeeze(0).cpu())
    return torch.stack(H).numpy(), torch.stack(Z).numpy()


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def knn_cos(test, mem, k=K):
    d = 1.0 - l2(test) @ l2(mem).T
    return np.partition(d, k - 1, axis=1)[:, :k].mean(1)


def knn_euc(test, mem, k=K):
    d = pairwise_distances(test, mem)
    return np.partition(d, k - 1, axis=1)[:, :k].mean(1)


def raw_feats(q):
    X = torch.cat([sn.x for sn in q["snapshots"]]); E = torch.cat([sn.edge_attr for sn in q["snapshots"]])
    return np.r_[X.mean(0).numpy(), E.mean(0).numpy(), np.log1p(X.shape[0]), np.log1p(E.shape[0])]


def attack_ratio(q):
    tot = sum(len(sn.edge_y) for sn in q["snapshots"])
    att = sum(int((sn.edge_y == 1).sum()) for sn in q["snapshots"])
    return att / max(tot, 1)


def probe(F, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    s = cross_val_predict(clf, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                          method="decision_function")
    return float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=os.path.join(ROOT, "graph_validation_v2.pt"))
    a = ap.parse_args()
    tr_seqs = torch.load(os.path.join(ROOT, "graph.pt"), map_location="cpu", weights_only=False)["sequences"]
    va = torch.load(a.val, map_location="cpu", weights_only=False)["sequences"]
    y = np.array([int(s["label"]) for s in va])
    perm = torch.randperm(len(tr_seqs), generator=torch.Generator().manual_seed(SEED)).tolist()
    mem_seqs = [tr_seqs[i] for i in perm[: int(0.9 * len(tr_seqs))]]
    print(f"file: {a.val}  sequences: {len(va)}  memory (graph.pt train): {len(mem_seqs)}")

    sec("Computing embeddings and raw features (a few minutes)")
    models = load_models()
    Hm, Zm = embed(models, mem_seqs); Hv, Zv = embed(models, va)
    Rm = np.stack([raw_feats(q) for q in mem_seqs]); Rv = np.stack([raw_feats(q) for q in va])
    ratio = np.array([attack_ratio(q) for q in va]); edges = np.expm1(Rv[:, -1])
    os.makedirs(os.path.join(ROOT, "diagnostics"), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    np.savez_compressed(os.path.join(ROOT, "diagnostics", f"embeddings_{stamp}.npz"),
                        Hm=Hm, Zm=Zm, Hv=Hv, Zv=Zv, y=y, edges=edges, ratio=ratio)

    # same split as verify_new_validation.py
    rng = np.random.default_rng(SEED)
    b = rng.permutation(np.where(y == 0)[0]); atk = np.where(y == 1)[0]
    tb = b[len(b) // 2:]; te = np.r_[tb, atk]; yt = y[te]
    print(f"test: benign {len(tb)}  attack {len(atk)}")
    print("attack-flow ratio among attack windows, quantiles 5/25/50/75/95 (%):",
          (100 * np.percentile(ratio[atk], [5, 25, 50, 75, 95])).round(2).tolist())

    scores = {"h": knn_cos(Hv, Hm), "z": knn_cos(Zv, Zm)}
    mu, sd = Rm[:, :-2].mean(0), Rm[:, :-2].std(0) + 1e-6
    scores["raw_mean_features"] = knn_euc((Rv[:, :-2] - mu) / sd, (Rm[:, :-2] - mu) / sd)
    OUT["base_auc"] = {k: float(roc_auc_score(yt, v[te])) for k, v in scores.items()}

    sec("1. DILUTION TEST: AUC using only attack windows with >= r share of attack flows")
    print(f"{'min ratio':>10} {'n_attack':>9} " + " ".join(f"{k:>18}" for k in scores))
    OUT["ratio_auc"] = {}
    for r in (0.0, 0.05, 0.2, 0.5, 0.8):
        sel = atk[ratio[atk] >= r]
        if len(sel) < 20:
            continue
        row = {k: float(roc_auc_score(np.r_[np.zeros(len(tb)), np.ones(len(sel))],
                                      np.r_[v[tb], v[sel]])) for k, v in scores.items()}
        OUT["ratio_auc"][str(r)] = dict(n=int(len(sel)), **row)
        print(f"{r:>10.2f} {len(sel):>9d} " + " ".join(f"{row[k]:>18.3f}" for k in scores))
    print("If AUC rises well above 0.5 as the ratio grows, label dilution explains most of the failure.")

    sec("2. SIZE CONTROL: AUC within 10 edge-count bins (weighted)")
    et = edges[te]; qs = np.quantile(et, np.linspace(0, 1, 11)); OUT["size_matched_auc"] = {}
    for k, v in scores.items():
        aucs, ws = [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            mk = (et >= lo) & (et <= hi)
            if (yt[mk] == 0).sum() >= 8 and (yt[mk] == 1).sum() >= 8:
                aucs.append(roc_auc_score(yt[mk], v[te][mk])); ws.append(int(mk.sum()))
        res = float(np.average(aucs, weights=ws)) if aucs else None
        OUT["size_matched_auc"][k] = res
        print(f"{k:>20}: {res if res is None else round(res, 4)}   (bins used: {len(aucs)}/10)")
        print(f"{'':>20}  spearman(score, edges) = {spearmanr(v[te], et)[0]:.3f}")

    sec("3. LINEAR PROBES (diagnostic only, uses labels, 5-fold CV over all validation sequences)")
    size = Rv[:, -2:]
    sets = {"size only": size, "raw mean features": Rv[:, :-2], "raw + size": Rv, "h": Hv, "z": Zv,
            "h + size": np.c_[Hv, size], "z + size": np.c_[Zv, size]}
    OUT["probe_auc"] = {}
    for k, F in sets.items():
        OUT["probe_auc"][k] = probe(F, y)
        print(f"{k:>20}: CV AUC = {OUT['probe_auc'][k]:.4f}")
    print("Reading: h/z well above 'size only' and near 'raw' => information is in the embedding and the kNN readout "
          "is the weak part.\n         h/z not above 'size only' => the encoder keeps little beyond graph size.")

    sec("4. kNN BASELINE ON RAW FEATURES vs EMBEDDINGS (memory = graph.pt train, same test set)")
    for k, v in OUT["base_auc"].items():
        print(f"{k:>20}: AUC = {v:.4f}")

    p = os.path.join(ROOT, "diagnostics", f"signal_{stamp}.json")
    json.dump(OUT, open(p, "w"), indent=2, default=float)
    print("\nSaved:", p)


if __name__ == "__main__":
    main()