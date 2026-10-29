r"""
sup_probe.py (v2) -- SUPERVISED baseline with a leakage-safe CHRONOLOGICAL split. Compares frozen h against graph size and
(optionally) plain raw-feature statistics, under the SAME protocol. CPU only.

Put in:  <PROJECT_ROOT>\training\sup_probe.py   (replace the old one)
Run:     python training\sup_probe.py --records member3_package_20261028_033449\member2_records.csv --val graph_validation_v2.pt *> sup_probe_console.txt
         (--val is optional; without it the raw-feature comparison is skipped)
Writes:  <PROJECT_ROOT>\diagnostics\sup_probe_<timestamp>.json (new file)

THIS IS A SUPERVISED RESULT (uses attack labels). Report it as: supervised readout, in-domain, chronological split.

Protocol
  * Dataset 2 windows sorted by start time; first 60% TRAIN, next 20% DEV, last 20% TEST
  * PURGE: windows that start before the previous block's last end_time are dropped
  * C (regularisation) is chosen on DEV by AUC; the decision threshold is the DEV point that maximises balanced accuracy
    (Youden's J) -- deliberately NOT max-F1, because attack share differs between blocks and max-F1 drifts towards "flag everything"
  * TEST is used once. F1 is printed next to the trivial "flag everything" F1, which any real result must beat
  * Metrics that do not depend on attack share (AUC, balanced accuracy) matter more than F1 here
"""
import os, sys, csv, json, time, argparse, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
                             balanced_accuracy_score, roc_curve)

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42


def ts(s):
    """parse a timestamp; handles ISO strings and numeric epochs (s / ms / us / ns, guessed from magnitude)"""
    s = str(s).strip()
    try:
        v = float(s)
    except ValueError:
        v = None
    try:
        if v is not None:
            a = abs(v); unit = "ns" if a > 1e17 else "us" if a > 1e14 else "ms" if a > 1e11 else "s"
            return np.datetime64(int(v), unit).astype("datetime64[s]")
        return np.datetime64(s.replace("Z", "")).astype("datetime64[s]")
    except Exception:
        sys.exit(f"cannot parse timestamp {s!r}; send me a few example values from the 'timestamp' column")


def bacc_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[int(np.argmax(tpr - fpr))])


def metrics(y, s, thr):
    pred = s >= thr
    return dict(auc=float(roc_auc_score(y, s)), ap=float(average_precision_score(y, s)), prevalence=float(y.mean()),
                f1=float(f1_score(y, pred)), f1_flag_everything=float(2 * y.mean() / (1 + y.mean())),
                precision=float(precision_score(y, pred, zero_division=0)), recall=float(recall_score(y, pred)),
                balanced_acc=float(balanced_accuracy_score(y, pred)), accuracy=float((pred == y).mean()), threshold=float(thr))


def fit_select(Xtr, ytr, Xdv, ydv):
    best = None
    for C in (0.001, 0.01, 0.1, 1.0, 10.0):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(Xtr, ytr)
        a = roc_auc_score(ydv, m.decision_function(Xdv))
        if best is None or a > best[0]:
            best = (a, C, m)
    return best


def raw_stats(seq):
    import torch
    X = torch.cat([g.x for g in seq["snapshots"]]); E = torch.cat([g.edge_attr for g in seq["snapshots"]])
    return np.r_[X.mean(0).numpy(), E.mean(0).numpy(), E.max(0).values.numpy(), E.std(0).numpy()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--records", required=True); ap.add_argument("--val", default=None)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.records, encoding="utf-8")))
    hcols = [c for c in rows[0] if c.startswith("h_")]
    rows = [r for r in rows if "validation" in r["dataset_id"] and r["label"] in ("0", "1")]
    st = np.array([ts(r["timestamp"]) for r in rows]); order = np.argsort(st, kind="stable"); rows = [rows[i] for i in order]
    n = len(rows); y = np.array([int(r["label"]) for r in rows])
    H = np.array([[float(r[c]) for c in hcols] for r in rows]); size = np.log1p(np.array([float(r["n_edges"]) for r in rows]))[:, None]
    st = np.array([ts(r["timestamp"]) for r in rows]); en = np.array([ts(r["end_time"]) if r["end_time"] else ts(r["timestamp"]) for r in rows])
    fam = np.array([r["attack_family"] for r in rows])
    print(f"rows: {n}  attack share {y.mean():.3f}\ntime span {st.min()} .. {st.max()}  (expected: 2018, CSE-CIC-IDS2018 capture)")
    if not (np.datetime64("2015") < st.min() < np.datetime64("2026")):
        print("WARNING: the parsed dates are not plausible; the unit guess may be wrong. Chronological order is still used, but send me example timestamp values.")

    feats = {"h": H, "size only": size, "h + size": np.c_[H, size]}
    if a.val:
        import torch
        vas = torch.load(a.val, map_location="cpu", weights_only=False)["sequences"]
        R = np.array([raw_stats(vas[int(r["source_index"])]) for r in rows])
        feats["raw stats"] = R; feats["raw stats + size"] = np.c_[R, size]
        print(f"raw-feature statistics computed from {a.val}: {R.shape[1]} columns (mean x, mean/max/std of edge features)")

    cuts = [int(0.6 * n), int(0.8 * n)]
    blk = np.zeros(n, int); blk[cuts[0]:] = 1; blk[cuts[1]:] = 2
    keep = np.ones(n, bool)
    for b in (1, 2):
        keep &= ~((blk == b) & (st < en[blk == b - 1].max()))
    names = ["TRAIN", "DEV", "TEST"]
    print("\nchronological blocks after purging time-overlapping windows:")
    for b in range(3):
        m = keep & (blk == b)
        print(f"  {names[b]:5s} n={int(m.sum()):5d} attack share={y[m].mean():.3f}  {st[m].min()} .. {st[m].max()}")
        u, c = np.unique(fam[m & (y == 1)], return_counts=True)
        print("        families: " + ", ".join(f"{str(k)}={int(v)}" for k, v in zip(u, c)))
    tr, dv, te = (keep & (blk == b) for b in range(3))
    for nm, m in (("TRAIN", tr), ("DEV", dv), ("TEST", te)):
        if len(np.unique(y[m])) < 2:
            sys.exit(f"{nm} block contains only one class")
    fam_tr = {x for f in fam[tr & (y == 1)] for x in f.split("|")}
    unseen = sorted({x for f in fam[te & (y == 1)] for x in f.split("|")} - fam_tr)
    print(f"\nattack families in TEST that never appear in TRAIN: {unseen if unseen else 'none'}")

    out = {"n": n, "blocks": {nm: int(m.sum()) for nm, m in zip(names, (tr, dv, te))}, "unseen_test_families": unseen,
           "chrono": {}, "random_optimistic": {}}
    print("\n" + "=" * 78 + "\nCHRONOLOGICAL SPLIT (report this)\n" + "=" * 78)
    for k, F in feats.items():
        auc, C, m = fit_select(F[tr], y[tr], F[dv], y[dv])
        thr = bacc_threshold(y[dv], m.decision_function(F[dv]))
        s_te = m.decision_function(F[te]); r = metrics(y[te], s_te, thr); r["C"] = C; r["dev_auc"] = float(auc)
        pred = s_te >= thr
        r["recall_by_family"] = {f: float(pred[(fam[te] == f)].mean()) for f in sorted(set(fam[te & (y == 1)])) if (fam[te] == f).sum() >= 10}
        r["benign_specificity"] = float((~pred[y[te] == 0]).mean())
        out["chrono"][k] = r
        print(f"{k:17s} AUC={r['auc']:.3f} AP={r['ap']:.3f} (prev {r['prevalence']:.3f}) | F1={r['f1']:.3f} (flag-everything F1 {r['f1_flag_everything']:.3f}) "
              f"| bal-acc={r['balanced_acc']:.3f} specificity={r['benign_specificity']:.3f} recall={r['recall']:.3f} (C={C}, dev AUC {auc:.3f})")
        print(f"{'':17s} recall by family (n>=10): " + ", ".join(f"{f}={v:.2f}" for f, v in r["recall_by_family"].items()))

    print("\n" + "=" * 78 + "\nRANDOM STRATIFIED SPLIT (optimistic -- neighbouring windows leak; do NOT report)\n" + "=" * 78)
    idx = np.arange(n)
    itr, ite = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    itr, idv = train_test_split(itr, test_size=0.25, stratify=y[itr], random_state=SEED)
    for k, F in feats.items():
        auc, C, m = fit_select(F[itr], y[itr], F[idv], y[idv])
        thr = bacc_threshold(y[idv], m.decision_function(F[idv]))
        r = metrics(y[ite], m.decision_function(F[ite]), thr); out["random_optimistic"][k] = r
        print(f"{k:17s} AUC={r['auc']:.3f}  F1={r['f1']:.3f}  bal-acc={r['balanced_acc']:.3f}")
    print("\nREADING: compare h + size against size only, and against raw stats. F1 only counts if it beats the flag-everything F1.")
    p = os.path.join(ROOT, "diagnostics", f"sup_probe_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(out, open(p, "w"), indent=2, default=float); print("\nSaved:", p)


if __name__ == "__main__":
    main()