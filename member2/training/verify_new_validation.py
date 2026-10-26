r"""
verify_new_validation.py -- READ-ONLY check of a rebuilt graph_validation.pt, then a first
valid evaluation of h and z.

Put in:  <PROJECT_ROOT>\training\verify_new_validation.py
Run:     python training\verify_new_validation.py *> verify_console.txt
Optional: --val <path to the new file>   (default: <PROJECT_ROOT>\graph_validation.pt)
Writes:  <PROJECT_ROOT>\diagnostics\verify_<timestamp>.json (new file each run)

Part 1 decides PASS/FAIL on the file itself (scale, shift, duplicates).
Part 2 only runs if Part 1 passes (override with --force). The threshold comes from
benign calibration data only; no test labels are used to choose it.
"""
import os, sys, json, time, argparse
from collections import Counter
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

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
    return gat, tr, pr, ck


@torch.no_grad()
def embed(models, seqs):
    gat, tr, pr = models
    H, Z = [], []
    for s in seqs:
        toks = []
        for g in s["snapshots"]:
            x = g.x.to(DEV).clamp(-5, 5); ea = g.edge_attr.to(DEV).clamp(-5, 5)   # training parity
            toks.append(global_mean_pooling(gat(x, g.edge_index.to(DEV), ea)))
        h = tr(torch.cat(toks, 0).unsqueeze(0))
        H.append(h.squeeze(0).cpu()); Z.append(pr(h).squeeze(0).cpu())
    return torch.stack(H).numpy(), torch.stack(Z).numpy()


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def knn(test, mem, k=K):
    d = 1.0 - l2(test) @ l2(mem).T
    return np.partition(d, k - 1, axis=1)[:, :k].mean(1)


def sizes(seqs):
    return (np.array([sum(s_.x.shape[0] for s_ in q["snapshots"]) for q in seqs], float),
            np.array([sum(s_.edge_index.shape[1] for s_ in q["snapshots"]) for q in seqs], float))


def feats(seqs, limit=500):
    X = torch.cat([sn.x for q in seqs[:limit] for sn in q["snapshots"]]).numpy()
    E = torch.cat([sn.edge_attr for q in seqs[:limit] for sn in q["snapshots"]]).numpy()
    return X, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=os.path.join(ROOT, "graph_validation_v2.pt"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    tr_seqs = torch.load(os.path.join(ROOT, "graph.pt"), map_location="cpu", weights_only=False)["sequences"]
    va = torch.load(a.val, map_location="cpu", weights_only=False)["sequences"]
    y = np.array([int(s["label"]) for s in va])
    types = [tuple(s.get("attack_types") or []) if s["label"] == 1 else ("BENIGN",) for s in va]
    print(f"file: {a.val}\nsequences: {len(va)}  labels: {dict(Counter(y.tolist()))}")
    print("attack types:", Counter(t for tt in types for t in tt).most_common(20))

    sec("PART 1A: feature scale vs graph.pt")
    Xt, Et = feats(tr_seqs); Xv, Ev = feats([s for s, l in zip(va, y) if l == 0] or va)
    bad = []
    for nm, At, Av in (("edge_attr", Et, Ev), ("node_x", Xt, Xv)):
        tmax = At.max(0); vmax = Av.max(0); vp99 = np.percentile(Av, 99, axis=0)
        tp50 = np.median(At, 0); vp50 = np.median(Av, 0)
        flag = (vmax > 5) | (vp99 > 1.5 * np.maximum(tmax, 1e-3))
        print(f"{nm}: train max overall={At.max():.3f}  valid max overall={Av.max():.3f}  "
              f"frac|v|>5 valid={(np.abs(Av) > 5).mean():.5f}  features flagged={int(flag.sum())}/53")
        for i in np.where(flag)[0][:10]:
            print(f"   feature {i}: train max={tmax[i]:.3f}  valid p99={vp99[i]:.3f} max={vmax[i]:.3f}")
        big_med = np.abs(np.log10((vp50 + 1e-3) / (tp50 + 1e-3))) > 1.0
        print(f"   features whose median differs >10x from train: {int(big_med.sum())}")
        bad.append(int(flag.sum()))
    scale_ok = max(bad) <= 3
    OUT["scale_flagged"] = bad

    sec("PART 1B: duplicates and sizes")
    from hashlib import md5
    def hs(q):
        h = md5()
        for s in q["snapshots"]:
            h.update(f"{s.x.shape}{float(s.x.sum()):.4f}{float(s.edge_attr.sum()):.4f}".encode())
        return h.hexdigest()
    th = {hs(q) for q in tr_seqs}; vh = [hs(q) for q in va]
    dup = sum(h in th for h in vh)
    print(f"validation sequences identical to a graph.pt sequence: {dup}; duplicates inside file: {len(vh) - len(set(vh))}")
    n, e = sizes(va)
    print(f"edges q05/50/95  benign={np.percentile(e[y==0],[5,50,95]).round(0)}  attack={np.percentile(e[y==1],[5,50,95]).round(0)}")
    print(f"size-only AUC (edges) = {roc_auc_score(y, e):.4f}   near-empty benign (<30 nodes) = {(n[y==0] < 30).mean():.3f}")
    OUT["dups_with_train"] = int(dup); OUT["size_only_auc"] = float(roc_auc_score(y, e))

    sec("PART 1C: embedding shift (graph.pt held-out benign vs validation benign)")
    models = load_models()[:3]
    perm = torch.randperm(len(tr_seqs), generator=torch.Generator().manual_seed(SEED)).tolist()
    cut = int(0.9 * len(tr_seqs))
    Htr, Ztr = embed(models, [tr_seqs[i] for i in perm[:cut]])
    Hho, Zho = embed(models, [tr_seqs[i] for i in perm[cut:]])
    Hva, Zva = embed(models, va)
    shift = {}
    for nm, (M, HO, V) in (("h", (Htr, Hho, Hva)), ("z", (Ztr, Zho, Zva))):
        s_ho = knn(HO, M); s_vb = knn(V[y == 0], M)
        shift[nm] = float(roc_auc_score(np.r_[np.zeros(len(s_ho)), np.ones(len(s_vb))], np.r_[s_ho, s_vb]))
        print(f"{nm}: shift AUC = {shift[nm]:.4f} (0.5 = same distribution; before the fix h was 0.995)  "
              f"median kNN dist heldout={np.median(s_ho):.3e} val-benign={np.median(s_vb):.3e}")
    OUT["shift_auc"] = shift
    shift_ok = shift["h"] < 0.65

    sec("PART 1 VERDICT")
    print(f"scale check : {'PASS' if scale_ok else 'FAIL'} (flagged features per tensor: {bad}; need <= 3)")
    print(f"shift check : {'PASS' if shift_ok else 'FAIL'} (h shift AUC {shift['h']:.3f}; need < 0.65)")
    print(f"duplicates  : {'PASS' if dup == 0 else 'WARN'}")
    OUT["verdict"] = dict(scale=scale_ok, shift=shift_ok, dups=dup == 0)
    if not (scale_ok and shift_ok) and not a.force:
        print("\nFile is NOT yet equivalent to graph.pt. Part 2 skipped (use --force to run anyway).")
        save(); return

    sec("PART 2: evaluation (memory = graph.pt training embeddings; threshold = benign calibration p95)")
    rng = np.random.default_rng(SEED)
    b = rng.permutation(np.where(y == 0)[0]); atk = np.where(y == 1)[0]
    cal, tb = b[: len(b) // 2], b[len(b) // 2:]
    te = np.r_[tb, atk]; yt = y[te]
    res = {}
    for nm, (M, V) in (("h", (Htr, Hva)), ("z", (Ztr, Zva))):
        s_all = knn(V, M); s_te = s_all[te]
        thr = float(np.percentile(s_all[cal], 95))
        pred = s_te >= thr
        r = dict(auc=float(roc_auc_score(yt, s_te)), ap=float(average_precision_score(yt, s_te)),
                 prevalence=float(yt.mean()), thr=thr, fpr=float(pred[yt == 0].mean()),
                 tpr=float(pred[yt == 1].mean()))
        r["balanced_acc"] = 0.5 * (r["tpr"] + 1 - r["fpr"])
        r["auc_negated_DIAGNOSTIC_ONLY"] = 1 - r["auc"]
        # size-stratified AUC
        et = e[te]; edges = np.quantile(et, np.linspace(0, 1, 6)); aucs, ws, per_bin = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mk = (et >= lo) & (et <= hi)
            if (yt[mk] == 0).sum() >= 10 and (yt[mk] == 1).sum() >= 10:
                aucs.append(roc_auc_score(yt[mk], s_te[mk])); ws.append(int(mk.sum()))
                per_bin.append(dict(lo=float(lo), hi=float(hi), n=int(mk.sum()), auc=float(aucs[-1])))
        r["size_stratified_auc"] = float(np.average(aucs, weights=ws)) if aucs else None
        r["size_bins"] = per_bin
        r["spearman_score_vs_edges"] = float(spearmanr(s_te, et)[0])
        per = {}
        for t in sorted({t for i in atk for t in types[i]}):
            idx = np.array([i for i in atk if t in types[i]])
            if len(idx) >= 20:
                per[t] = dict(n=int(len(idx)), auc=float(roc_auc_score(
                    np.r_[np.zeros(len(tb)), np.ones(len(idx))], np.r_[s_all[tb], s_all[idx]])))
        r["per_attack_auc_vs_benign"] = per
        res[nm] = r
        print(f"\n[{nm}] AUC={r['auc']:.4f} AP={r['ap']:.4f} (prevalence {r['prevalence']:.3f})  "
              f"size-stratified AUC={r['size_stratified_auc']}")
        print(f"    threshold(benign p95)={thr:.4e}  FPR={r['fpr']:.3f} TPR={r['tpr']:.3f} balanced acc={r['balanced_acc']:.3f}")
        print(f"    spearman(score, edges)={r['spearman_score_vs_edges']:.3f}")
        print("    per-attack AUC:", {k: round(v['auc'], 3) for k, v in per.items()})
    OUT["eval"] = res
    save()


def save():
    os.makedirs(os.path.join(ROOT, "diagnostics"), exist_ok=True)
    p = os.path.join(ROOT, "diagnostics", f"verify_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(OUT, open(p, "w"), indent=2, default=float)
    print("\nSaved:", p)


if __name__ == "__main__":
    main()