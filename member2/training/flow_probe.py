r"""
flow_probe.py -- SUPERVISED, FLOW-LEVEL feasibility check (CPU). Are individual attack flows detectable, and does flagging
windows by their most suspicious flows beat the window-level classifiers?

Put in:  <PROJECT_ROOT>\training\flow_probe.py   (needs sup_common.py next to it)
Run:     python training\flow_probe.py --records member3_package_20261028_033449\member2_records.csv --val graph_validation_v2.pt *> flow_console.txt
Options: --with-nodes   also use the source/destination node features of each flow (159 features instead of 53)
Writes:  <PROJECT_ROOT>\diagnostics\flow_<timestamp>.json (new file)

Uses the per-flow label edge_y stored in graph_validation_v2.pt and the SAME chronological folds as sup_folds.py
(fit = first 80% of earlier chunks, calib = last 20%, test = next chunk). Flows are sampled per window to keep memory small:
fit 300, calib 500, test 1000 flows per window (uniform random, seed 42).
  * flow-level threshold = 99th percentile of BENIGN flow scores on CALIB windows (a fixed 1% false-positive rate;
    attack share does not influence it)
  * window score = 99th percentile / max of that window's flow scores; window threshold = Youden on CALIB windows
SUPERVISED, in-domain, chronological. Not zero-day detection.
"""
import os, sys, json, time, argparse, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.append(HERE)
from sup_common import load_rows, make_folds, youden

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(HERE)
N_FIT, N_CAL, N_TEST, SEED = 300, 500, 1000, 42
REF_H_SIZE_AUC = {2: 0.890, 3: 0.675, 4: 0.890, 5: 0.796}     # window-level h + size, from folds_console.txt


def window_flows(seq, n, rng, with_nodes):
    Xs, Ys = [], []
    for g in seq["snapshots"]:
        ea = np.asarray(g.edge_attr, dtype=np.float32); yv = (np.asarray(g.edge_y) == 1).astype(np.int8)
        if with_nodes:
            ei = np.asarray(g.edge_index); x = np.asarray(g.x, dtype=np.float32)
            ea = np.concatenate([ea, x[ei[0]], x[ei[1]]], axis=1)
        Xs.append(ea); Ys.append(yv)
    X = np.concatenate(Xs); y = np.concatenate(Ys)
    idx = rng.choice(len(y), min(n, len(y)), replace=False)
    return X[idx], y[idx]


def gather(vas, D, widx, n, rng, with_nodes):
    X, y, w = [], [], []
    for i in widx:
        a, b = window_flows(vas[int(D["src"][i])], n, rng, with_nodes)
        X.append(a); y.append(b); w.append(np.full(len(b), i))
    return np.concatenate(X), np.concatenate(y), np.concatenate(w)


def agg(scores, w, widx, how):
    out = []
    for i in widx:
        s = scores[w == i]
        out.append(np.percentile(s, 99) if how == "p99" else s.max())
    return np.array(out)


def run(D, vas, with_nodes=False):
    yw = D["y"]; res = {}
    for f in make_folds(D):
        if not f["ok"]:
            continue
        k = f["k"]; rng = np.random.default_rng(SEED + k)
        Xf, yf, _ = gather(vas, D, f["fit"], N_FIT, rng, with_nodes)
        Xc, yc, wc = gather(vas, D, f["cal"], N_CAL, rng, with_nodes)
        Xt, yt, wt = gather(vas, D, f["test"], N_TEST, rng, with_nodes)
        print(f"\nfold {k}: flows fit {len(yf):,} (attack {yf.mean():.4f})  calib {len(yc):,}  test {len(yt):,} (attack {yt.mean():.4f})  "
              f"windows fit/calib/test {len(f['fit'])}/{len(f['cal'])}/{len(f['test'])}")
        if yt.sum() < 20 or yf.sum() < 20:
            print("   too few attack flows in this fold; skipped"); continue
        models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")),
                  "hist-gradient-boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, class_weight="balanced", random_state=SEED)}
        res[k] = {}
        for name, m in models.items():
            m.fit(Xf, yf)
            sc = lambda X: m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]
            s_c, s_t = sc(Xc), sc(Xt)
            thr_f = float(np.percentile(s_c[yc == 0], 99))                       # fixed 1% benign-flow false-positive rate
            r = dict(flow_auc=float(roc_auc_score(yt, s_t)), flow_ap=float(average_precision_score(yt, s_t)),
                     flow_attack_share=float(yt.mean()), flow_tpr_at_1pct_fpr=float((s_t[yt == 1] >= thr_f).mean()),
                     flow_fpr_realised=float((s_t[yt == 0] >= thr_f).mean()))
            for how in ("p99", "max"):
                a_c = agg(s_c, wc, f["cal"], how); a_t = agg(s_t, wt, f["test"], how)
                thr = youden(yw[f["cal"]], a_c); pred = a_t >= thr; yy = yw[f["test"]]
                rec = float(pred[yy == 1].mean()); spec = float((~pred[yy == 0]).mean())
                r[f"window_{how}"] = dict(auc=float(roc_auc_score(yy, a_t)), bal_acc=0.5 * (rec + spec), recall=rec, specificity=spec)
            res[k][name] = r
            w99 = r["window_p99"]
            print(f"  {name:22s} FLOW: AUC={r['flow_auc']:.3f} AP={r['flow_ap']:.3f} (attack share {r['flow_attack_share']:.4f})  "
                  f"TPR@1%FPR={r['flow_tpr_at_1pct_fpr']:.3f}   WINDOW (p99 of flow scores): AUC={w99['auc']:.3f} "
                  f"bal-acc={w99['bal_acc']:.3f} spec={w99['specificity']:.2f} recall={w99['recall']:.2f}")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--records", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--with-nodes", action="store_true")
    a = ap.parse_args()
    import torch
    D = load_rows(a.records)
    vas = torch.load(a.val, map_location="cpu", weights_only=False)["sequences"]
    print(f"windows {len(D['y'])}  features per flow: {159 if a.with_nodes else 53}")
    res = run(D, vas, a.with_nodes)
    print("\n" + "=" * 90 + "\nSUMMARY (mean over folds)\n" + "=" * 90)
    for name in ("logistic", "hist-gradient-boosting"):
        rows = [res[k][name] for k in res]
        if not rows:
            continue
        mean = lambda g: float(np.mean([g(r) for r in rows]))
        print(f"{name:22s} flow AUC {mean(lambda r: r['flow_auc']):.3f} | flow AP {mean(lambda r: r['flow_ap']):.3f} | "
              f"TPR@1%FPR {mean(lambda r: r['flow_tpr_at_1pct_fpr']):.3f} | window AUC {mean(lambda r: r['window_p99']['auc']):.3f} | "
              f"window bal-acc {mean(lambda r: r['window_p99']['bal_acc']):.3f}")
    ref = [REF_H_SIZE_AUC[k] for k in res if k in REF_H_SIZE_AUC]
    print(f"\nREFERENCE window-level 'h + size' (same folds): mean AUC {np.mean(ref):.3f}, mean balanced accuracy 0.678 "
          f"(per-fold AUC {[REF_H_SIZE_AUC[k] for k in res if k in REF_H_SIZE_AUC]})")
    print("READING: if flow AUC / TPR@1%FPR are high and window AUC / balanced accuracy beat the reference, flow-level detection is the "
          "better route. If flows are barely separable, the labels, not the model, are the limit.")
    p = os.path.join(ROOT, "diagnostics", f"flow_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(res, open(p, "w"), indent=2, default=float); print("\nSaved:", p)


if __name__ == "__main__":
    main()