r"""
sup_common.py -- shared helpers for sup_folds.py and finetune_sup.py (chronological folds, metrics, bootstrap).
Put in: <PROJECT_ROOT>\training\sup_common.py

FOLD DESIGN (rolling origin, leakage-safe)
  Dataset 2 windows are sorted by start time and cut into 6 equal chunks. For test chunk k = 2..5 (0-based):
    all earlier chunks = TRAIN-AVAILABLE; its first 80% = FIT, last 20% = CALIB (used to pick C / epochs and the threshold)
    TEST = chunk k.  Windows that start before the previous block's last end_time are dropped (no time overlap).
  The same folds, the same CALIB rule and the same threshold rule (Youden: max TPR - FPR on CALIB) are used for EVERY method.
"""
import csv, sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve

N_CHUNKS, FIRST_TEST = 6, 2


def ts(s):
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
        sys.exit(f"cannot parse timestamp {s!r}")


def load_rows(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    hcols = [c for c in rows[0] if c.startswith("h_")]
    rows = [r for r in rows if "validation" in r["dataset_id"] and r["label"] in ("0", "1")]
    st = np.array([ts(r["timestamp"]) for r in rows]); order = np.argsort(st, kind="stable"); rows = [rows[i] for i in order]
    st = st[order]
    en = np.array([ts(r["end_time"]) if r["end_time"] else s for r, s in zip(rows, st)])
    return dict(rows=rows, st=st, en=en, y=np.array([int(r["label"]) for r in rows]),
                H=np.array([[float(r[c]) for c in hcols] for r in rows]),
                size=np.log1p(np.array([float(r["n_edges"]) for r in rows]))[:, None],
                fam=np.array([r["attack_family"] for r in rows]), src=np.array([int(r["source_index"]) for r in rows]))


def make_folds(D):
    n = len(D["y"]); edges = np.linspace(0, n, N_CHUNKS + 1).astype(int); folds = []
    for k in range(FIRST_TEST, N_CHUNKS):
        tr_all = np.arange(0, edges[k]); te = np.arange(edges[k], edges[k + 1])
        cut = int(0.8 * len(tr_all)); fit = tr_all[:cut]; cal = tr_all[cut:]
        cal = cal[D["st"][cal] >= D["en"][fit].max()]
        te = te[D["st"][te] >= D["en"][np.r_[fit, cal]].max()]
        ok = all(len(np.unique(D["y"][ix])) == 2 for ix in (fit, cal, te))
        folds.append(dict(k=k, fit=fit, cal=cal, test=te, ok=ok))
    return folds


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[int(np.argmax(tpr - fpr))])


def metrics(y, s, thr):
    pred = s >= thr; p = y.mean()
    rec = float(pred[y == 1].mean()); spec = float((~pred[y == 0]).mean())
    return dict(auc=float(roc_auc_score(y, s)), ap=float(average_precision_score(y, s)), prevalence=float(p),
                f1=float(f1_score(y, pred)), f1_flag_all=float(2 * p / (1 + p)), recall=rec, specificity=spec,
                bal_acc=0.5 * (rec + spec))


def fold_stat(y, s, thr, name):
    if name == "auc":
        return roc_auc_score(y, s)
    pred = s >= thr
    return 0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())


def boot_ci(y, s, thr, B=500, seed=0):
    rng = np.random.default_rng(seed); a, b = [], []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        a.append(fold_stat(y[i], s[i], thr, "auc")); b.append(fold_stat(y[i], s[i], thr, "bacc"))
    return np.percentile(a, [2.5, 97.5]), np.percentile(b, [2.5, 97.5])


def paired_diff(res, A, B_, stat, B=500, seed=0):
    """mean over folds of stat(A) - stat(B_), bootstrap over test windows (same resample for both methods)."""
    ks = [k for k in res[A] if k in res[B_]]
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        d = []
        for k in ks:
            y, sA, tA = res[A][k]; _, sB, tB = res[B_][k]
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) < 2:
                continue
            d.append(fold_stat(y[i], sA[i], tA, stat) - fold_stat(y[i], sB[i], tB, stat))
        if d:
            out.append(np.mean(d))
    return float(np.mean(out)), np.percentile(out, [2.5, 97.5])