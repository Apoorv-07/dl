r"""
score_readouts.py -- READ-ONLY, CPU only, NO model needed. Step 1: does a different anomaly SCORER
on the EXISTING embeddings (h, z) beat the current kNN-cosine (k=5)?

Put in:  <PROJECT_ROOT>\training\score_readouts.py
Run:     python training\score_readouts.py *> readouts_console.txt
         (uses the newest diagnostics\embeddings_*.npz written by diagnose_attack_signal.py;
          or pass --npz <file>)
Writes:  <PROJECT_ROOT>\diagnostics\readouts_<timestamp>.json (new file each run)

Protocol (no test-label leakage):
  * Every scorer is fitted ONLY on the benign graph.pt memory (Hm / Zm). No validation labels are used to fit.
  * The validation file is split 50/50 (stratified) into DEV and TEST, repeated over 20 random seeds.
  * The scorer is CHOSEN on DEV (best AUC), then REPORTED on TEST. TEST labels never choose anything.
  * The threshold (FPR/TPR) comes from the benign p95 of the DEV half only.
  * Scores are NEVER sign-flipped. If every scorer is below 0.5, that is reported as is.
Caveat: choosing a scorer on DEV uses DEV labels, so a "selected" result is a lightly supervised
        model-selection number, not a purely unsupervised one. The fixed baseline (knn-cos k=5) is shown beside it.
"""
import os, sys, glob, json, time, argparse, warnings
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def knn_cos(V, M, k):
    d = 1.0 - l2(V) @ l2(M).T
    return np.partition(d, k - 1, axis=1)[:, :k].mean(1)


def centroid_cos(V, M):
    c = l2(M).mean(0); c /= max(np.linalg.norm(c), 1e-12)
    return 1.0 - l2(V) @ c


def maha(V, M):
    lw = LedoitWolf().fit(M)
    return lw.mahalanobis(V)


def knn_std(V, M, k):
    mu, sd = M.mean(0), M.std(0) + 1e-8
    nn = NearestNeighbors(n_neighbors=k).fit((M - mu) / sd)
    return nn.kneighbors((V - mu) / sd)[0].mean(1)


def lof(V, M, k=20):
    m = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(M)
    return -m.score_samples(V)


def all_scores(V, M, tag):
    s = {}
    for k in (1, 5, 20, 50):
        s[f"{tag}:knn_cos_k{k}"] = knn_cos(V, M, k)
    s[f"{tag}:centroid_cos"] = centroid_cos(V, M)
    s[f"{tag}:mahalanobis"] = maha(V, M)
    for k in (5, 20):
        s[f"{tag}:knn_std_k{k}"] = knn_std(V, M, k)
    s[f"{tag}:lof_k20"] = lof(V, M, 20)
    return s


def split(y, seed):
    rng = np.random.default_rng(seed)
    dev, test = [], []
    for c in (0, 1):
        idx = rng.permutation(np.where(y == c)[0]); h = len(idx) // 2
        dev += list(idx[:h]); test += list(idx[h:])
    return np.array(dev), np.array(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None)
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    path = a.npz or max(glob.glob(os.path.join(ROOT, "diagnostics", "embeddings_*.npz")), key=os.path.getmtime)
    d = np.load(path)
    Hm, Zm, Hv, Zv, y, ratio = d["Hm"], d["Zm"], d["Hv"], d["Zv"], d["y"], d["ratio"]
    print(f"embeddings file: {path}\nmemory: {len(Hm)}   validation: {len(y)} (benign {(y==0).sum()}, attack {(y==1).sum()})")

    S = {**all_scores(Hv, Hm, "h"), **all_scores(Zv, Zm, "z")}
    names = list(S)
    R = a.repeats
    dev_auc = np.zeros((R, len(names))); test_auc = np.zeros((R, len(names)))
    test_auc_hi = np.zeros((R, len(names)))   # test benign vs attack windows with >=20% attack flows
    sel_rows = []
    for r in range(R):
        dev, te = split(y, 1000 + r)
        yd, yt = y[dev], y[te]
        hi = te[(y[te] == 0) | (ratio[te] >= 0.2)]
        for j, n in enumerate(names):
            dev_auc[r, j] = roc_auc_score(yd, S[n][dev])
            test_auc[r, j] = roc_auc_score(yt, S[n][te])
            test_auc_hi[r, j] = roc_auc_score(y[hi], S[n][hi])
        j = int(dev_auc[r].argmax()); n = names[j]
        thr = np.percentile(S[n][dev][yd == 0], 95); pred = S[n][te] >= thr
        sel_rows.append(dict(seed=1000 + r, selected=n, dev_auc=float(dev_auc[r, j]), test_auc=float(test_auc[r, j]),
                             test_ap=float(average_precision_score(yt, S[n][te])), prevalence=float(yt.mean()),
                             fpr=float(pred[yt == 0].mean()), tpr=float(pred[yt == 1].mean())))

    print(f"\nMean over {R} random DEV/TEST splits (std in brackets). No sign flipping.\n")
    print(f"{'scorer':22s} {'DEV auc':>14s} {'TEST auc':>14s} {'TEST auc (attack>=20% flows)':>30s}")
    for j, n in sorted(enumerate(names), key=lambda t: -test_auc[:, t[0]].mean()):
        print(f"{n:22s} {dev_auc[:, j].mean():8.3f} ({dev_auc[:, j].std():.3f}) {test_auc[:, j].mean():8.3f} ({test_auc[:, j].std():.3f}) "
              f"{test_auc_hi[:, j].mean():14.3f} ({test_auc_hi[:, j].std():.3f})")

    base = names.index("h:knn_cos_k5")
    print(f"\nCURRENT baseline h:knn_cos_k5   TEST AUC = {test_auc[:, base].mean():.3f}")
    sel_auc = np.array([s["test_auc"] for s in sel_rows])
    sel_names = [s["selected"] for s in sel_rows]
    top = max(set(sel_names), key=sel_names.count)
    print(f"DEV-selected scorer, TEST AUC  = {sel_auc.mean():.3f} (std {sel_auc.std():.3f})   most often selected: {top} "
          f"({sel_names.count(top)}/{R})")
    print(f"selected-scorer TEST: AP={np.mean([s['test_ap'] for s in sel_rows]):.3f} (prevalence {np.mean([s['prevalence'] for s in sel_rows]):.3f})  "
          f"FPR={np.mean([s['fpr'] for s in sel_rows]):.3f}  TPR={np.mean([s['tpr'] for s in sel_rows]):.3f}  (threshold = DEV-benign p95)")
    best = test_auc.mean(0).max()
    print("\nREADING:")
    print(" - If no scorer's TEST AUC is clearly above 0.5 (say > 0.6), the problem is the representation, not the readout")
    print("   -> go to step 2 (input transform) and step 3 (pooling), which need retraining.")
    print(f" - Best fixed scorer here: {names[int(test_auc.mean(0).argmax())]} at {best:.3f}.")
    print(" - A score below 0.5 means attacks look MORE normal than benign to that scorer. Do not flip it to claim a detector.")

    out = os.path.join(ROOT, "diagnostics", f"readouts_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(source=path, scorers=names, dev_auc_mean=dev_auc.mean(0).tolist(), test_auc_mean=test_auc.mean(0).tolist(),
                   test_auc_hi_mean=test_auc_hi.mean(0).tolist(), selected=sel_rows), open(out, "w"), indent=2)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()