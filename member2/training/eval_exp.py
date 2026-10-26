r"""
eval_exp.py -- READ-ONLY evaluation of any checkpoint (your current one, a new experiment, or an UNTRAINED encoder)
on graph_validation_v2.pt, using exactly the same protocol as verify_new_validation.py.

Put in:  <PROJECT_ROOT>\training\eval_exp.py   (needs exp_common.py next to it)
Examples:
  python training\eval_exp.py --label baseline --ckpt checkpoints_validation\best_model.pt
  python training\eval_exp.py --label random_init --random-init
  python training\eval_exp.py --label e1_log --ckpt checkpoints_exp\e1_log\best_model.pt
Add  --val <path>  if your validation file is elsewhere (default: <PROJECT_ROOT>\graph_validation_v2.pt).
Writes: diagnostics\eval_<label>_<timestamp>.json and diagnostics\embeddings_<label>_<timestamp>.npz (new files).

Protocol: memory = graph.pt training embeddings (first 90% after the seed-42 permutation); test set = second half of the
benign validation windows (seed 42, same as verify_new_validation.py) + all attack windows. No label is used to fit or choose
anything. Scores are never sign-flipped. The linear probe uses labels and is DIAGNOSTIC ONLY.
"""
import os, sys, json, time, argparse, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from exp_common import load_checkpoint, build_models, set_mode, embed, default_cfg

warnings.filterwarnings("ignore")
SEED, K = 42, 5

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--ckpt", default=None)
ap.add_argument("--random-init", action="store_true", help="evaluate an UNTRAINED encoder (seed 42) as a control")
ap.add_argument("--transform", choices=["none", "log"], default="none")
ap.add_argument("--pooling", choices=["mean", "meanmax"], default="mean")
ap.add_argument("--val", default=os.path.join(ROOT, "graph_validation_v2.pt"))
a = ap.parse_args()
if not a.random_init and not a.ckpt:
    sys.exit("give --ckpt <path> or --random-init")


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def knn(V, M, k=K):
    d = 1.0 - l2(V) @ l2(M).T
    return np.partition(d, k - 1, axis=1)[:, :k].mean(1)


def ratio_of(q):
    tot = sum(len(s.edge_y) for s in q["snapshots"]); att = sum(int((s.edge_y == 1).sum()) for s in q["snapshots"])
    return att / max(tot, 1)


def edges_of(q):
    return sum(s.edge_index.shape[1] for s in q["snapshots"])


if a.random_init:
    torch.manual_seed(SEED)
    cfg = dict(default_cfg(), transform=a.transform, pooling=a.pooling)
    m = build_models(cfg); set_mode(m, False); src = "UNTRAINED (random init, seed 42)"
else:
    m, cfg, ck = load_checkpoint(a.ckpt); src = f"{a.ckpt} (best epoch {ck.get('best_epoch')}, val loss {ck.get('best_validation_loss')})"
print(f"label: {a.label}\nmodel: {src}\ncfg: {cfg}\nvalidation file: {a.val}")

tr = torch.load(os.path.join(ROOT, "graph.pt"), map_location="cpu", weights_only=False)["sequences"]
va = torch.load(a.val, map_location="cpu", weights_only=False)["sequences"]
y = np.array([int(s["label"]) for s in va])
perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(SEED)).tolist()
mem = [tr[i] for i in perm[: int(0.9 * len(tr))]]
ho = [tr[i] for i in perm[int(0.9 * len(tr)):]]
print("embedding (a few minutes)...")
Hm, Zm = embed(m, mem, cfg); Hh, Zh = embed(m, ho, cfg); Hv, Zv = embed(m, va, cfg)
ratio = np.array([ratio_of(q) for q in va]); edges = np.array([edges_of(q) for q in va], float)

stamp = time.strftime("%Y%m%d_%H%M%S"); os.makedirs(os.path.join(ROOT, "diagnostics"), exist_ok=True)
np.savez_compressed(os.path.join(ROOT, "diagnostics", f"embeddings_{a.label}_{stamp}.npz"),
                    Hm=Hm, Zm=Zm, Hv=Hv, Zv=Zv, y=y, edges=edges, ratio=ratio)

rng = np.random.default_rng(SEED)
b = rng.permutation(np.where(y == 0)[0]); atk = np.where(y == 1)[0]
cal, tb = b[: len(b) // 2], b[len(b) // 2:]; te = np.r_[tb, atk]; yt = y[te]
hi = np.r_[tb, atk[ratio[atk] >= 0.2]]; yhi = y[hi]
out = dict(label=a.label, model=src, cfg=cfg, n_test_benign=int(len(tb)), n_test_attack=int(len(atk)))

print("\n" + "=" * 66 + "\nRESULTS (same test set and protocol as verify_new_validation.py)\n" + "=" * 66)
for nm, (M, HO, V) in (("h", (Hm, Hh, Hv)), ("z", (Zm, Zh, Zv))):
    sc = {"knn_cos_k5": knn(V, M)}
    sc["mahalanobis"] = LedoitWolf().fit(M).mahalanobis(V)
    shift = roc_auc_score(np.r_[np.zeros(len(HO)), np.ones((y == 0).sum())], np.r_[knn(HO, M), sc["knn_cos_k5"][y == 0]])
    r = dict(shift_auc=float(shift))
    print(f"\n[{nm}] benign-shift AUC (0.5 = same as pretraining) = {shift:.3f}   "
          f"pairwise-cos spread: memory mean cos = {float((l2(M[:500]) @ l2(M[:500]).T).mean()):.3f}")
    for sn, s in sc.items():
        s_te = s[te]; thr = float(np.percentile(s[cal], 95)); pred = s_te >= thr
        et = edges[te]; qs = np.quantile(et, np.linspace(0, 1, 6)); aucs, ws = [], []
        for lo, up in zip(qs[:-1], qs[1:]):
            mk = (et >= lo) & (et <= up)
            if (yt[mk] == 0).sum() >= 10 and (yt[mk] == 1).sum() >= 10:
                aucs.append(roc_auc_score(yt[mk], s_te[mk])); ws.append(int(mk.sum()))
        rr = dict(auc=float(roc_auc_score(yt, s_te)), ap=float(average_precision_score(yt, s_te)),
                  prevalence=float(yt.mean()), auc_attack_ge20pct=float(roc_auc_score(yhi, s[hi])),
                  size_stratified_auc=float(np.average(aucs, weights=ws)) if aucs else None,
                  fpr=float(pred[yt == 0].mean()), tpr=float(pred[yt == 1].mean()))
        r[sn] = rr
        print(f"  {sn:12s} AUC={rr['auc']:.3f}  AP={rr['ap']:.3f} (prev {rr['prevalence']:.3f})  "
              f"AUC(attack>=20% flows)={rr['auc_attack_ge20pct']:.3f}  size-stratified={rr['size_stratified_auc']}  "
              f"FPR={rr['fpr']:.3f} TPR={rr['tpr']:.3f}")
    out[nm] = r

size = np.log1p(edges)[:, None]
def probe(F):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    return float(roc_auc_score(y, cross_val_predict(clf, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                                      method="decision_function")))
out["probe_auc"] = dict(size_only=probe(size), h=probe(Hv), z=probe(Zv), h_plus_size=probe(np.c_[Hv, size]))
print("\nLinear probe, 5-fold CV (uses labels, DIAGNOSTIC ONLY): " + "  ".join(f"{k}={v:.3f}" for k, v in out["probe_auc"].items()))
print("\nREFERENCE (your current checkpoint, same protocol): h kNN AUC 0.360, z 0.427; raw-feature kNN 0.556 "
      "(0.970 on attack windows with >=50% attack flows); probes size-only 0.827, h 0.849, h+size 0.942.")
print("Below 0.5 = attacks look more normal than benign. Do not flip the sign to report a detector.")

p = os.path.join(ROOT, "diagnostics", f"eval_{a.label}_{stamp}.json")
json.dump(out, open(p, "w"), indent=2, default=float); print("\nSaved:", p)