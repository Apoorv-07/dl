r"""
sup_folds.py -- SUPERVISED multi-fold chronological evaluation (CPU). Frozen-feature probes, and (optionally) the fine-tuned
models written by finetune_sup.py, all under the SAME folds and threshold rule (see sup_common.py).

Put in:  <PROJECT_ROOT>\training\sup_folds.py   (needs sup_common.py next to it)
Run:     python training\sup_folds.py --records member3_package_20261028_033449\member2_records.csv --val graph_validation_v2.pt *> folds_console.txt
Later, after fine-tuning:   add   --ft "diagnostics\ft_*.npz"   to include the fine-tuned results.
Writes:  <PROJECT_ROOT>\diagnostics\folds_<timestamp>.json (new file)

SUPERVISED result (uses attack labels): report as "supervised, in-domain, rolling-origin chronological folds".
"""
import os, sys, glob, json, time, argparse, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.append(HERE)
from sup_common import load_rows, make_folds, youden, metrics, boot_ci, paired_diff

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(HERE)


def raw_stats(seq):
    import torch
    X = torch.cat([g.x for g in seq["snapshots"]]); E = torch.cat([g.edge_attr for g in seq["snapshots"]])
    return np.r_[X.mean(0).numpy(), E.mean(0).numpy(), E.max(0).values.numpy(), E.std(0).numpy()]


def fit_probe(F, y, fit, cal):
    best = None
    for C in (0.001, 0.01, 0.1, 1.0, 10.0):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(F[fit], y[fit])
        a = roc_auc_score(y[cal], m.decision_function(F[cal]))
        if best is None or a > best[0]:
            best = (a, C, m)
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--records", required=True)
    ap.add_argument("--val", default=None); ap.add_argument("--ft", default=None)
    ap.add_argument("--min-ratio", type=float, default=0.0,
                    help="label-definition sensitivity check: keep benign windows and only attack windows whose share of attack flows is >= this "
                         "(e.g. 0.05). Windows with fewer attack flows are EXCLUDED, not relabelled. Do not combine with --ft.")
    a = ap.parse_args()
    D = load_rows(a.records)
    if a.min_ratio > 0:
        ratio = np.array([float(r["attack_flow_ratio"]) if r.get("attack_flow_ratio", "") != "" else 0.0 for r in D["rows"]])
        keep = (D["y"] == 0) | (ratio >= a.min_ratio)
        print(f"--min-ratio {a.min_ratio}: excluded {int((~keep).sum())} low-ratio attack windows (of {int((D['y'] == 1).sum())}); "
              f"kept {int(keep.sum())}")
        D = {k: (v[keep] if isinstance(v, np.ndarray) else [x for x, kp in zip(v, keep) if kp]) for k, v in D.items()}
    y = D["y"]; n = len(y)
    print(f"rows {n}, attack share {y.mean():.3f}, time span {D['st'].min()} .. {D['st'].max()}")
    feats = {"size only": D["size"], "h": D["H"], "h + size": np.c_[D["H"], D["size"]]}
    if a.val:
        import torch
        vas = torch.load(a.val, map_location="cpu", weights_only=False)["sequences"]
        R = np.array([raw_stats(vas[i]) for i in D["src"]])
        feats["raw stats"] = R; feats["raw stats + size"] = np.c_[R, D["size"]]
    folds = make_folds(D)
    print("\nFOLDS (rolling origin; fit = first 80% of earlier chunks, calib = last 20%, test = next chunk)")
    for f in folds:
        fam_te = sorted({x for s in D["fam"][f["test"]][y[f["test"]] == 1] for x in s.split("|")})
        fam_fit = {x for s in D["fam"][f["fit"]][y[f["fit"]] == 1] for x in s.split("|")}
        print(f"  fold {f['k']}: fit {len(f['fit'])}  calib {len(f['cal'])}  test {len(f['test'])} (attack {y[f['test']].mean():.2f})  "
              f"{'OK' if f['ok'] else 'SKIPPED: a block lacks one class'}  unseen test families: {sorted(set(fam_te) - fam_fit)}")
    folds = [f for f in folds if f["ok"]]
    res = {}; out = {"folds": {}}
    for name, F in feats.items():
        res[name] = {}
        for f in folds:
            auc, C, m = fit_probe(F, y, f["fit"], f["cal"])
            thr = youden(y[f["cal"]], m.decision_function(F[f["cal"]]))
            res[name][f["k"]] = (y[f["test"]], m.decision_function(F[f["test"]]), thr)
    if a.ft:
        pos = {int(s): i for i, s in enumerate(D["src"])}
        for p in sorted(glob.glob(a.ft)):
            z = np.load(p, allow_pickle=True); lab = str(z["label"]); k = int(z["fold"])
            idx = np.array([pos[int(s)] for s in z["test_src"]])
            exp = next((f for f in folds if f["k"] == k), None)
            if exp is None or set(idx) != set(exp["test"]):
                print(f"WARNING: {p}: test windows differ from this script's fold {k}; skipped"); continue
            thr = youden(z["calib_y"], z["calib_scores"])
            res.setdefault(lab, {})[k] = (y[idx], z["test_scores"], thr)

    print("\n" + "=" * 100 + "\nPER-FOLD RESULTS (test chunk; threshold = Youden on CALIB; 95% bootstrap CI over test windows)\n" + "=" * 100)
    for name in res:
        print(f"\n[{name}]")
        out["folds"][name] = {}
        for k, (yt, s, thr) in sorted(res[name].items()):
            m = metrics(yt, s, thr); (la, ha), (lb, hb) = boot_ci(yt, s, thr)
            out["folds"][name][k] = m
            print(f"  fold {k}: AUC={m['auc']:.3f} [{la:.3f},{ha:.3f}]  bal-acc={m['bal_acc']:.3f} [{lb:.3f},{hb:.3f}]  "
                  f"F1={m['f1']:.3f} (flag-all {m['f1_flag_all']:.3f})  spec={m['specificity']:.2f} recall={m['recall']:.2f}  n={len(yt)}")
    print("\n" + "=" * 100 + "\nMEAN OVER FOLDS\n" + "=" * 100)
    print(f"{'method':22s} {'AUC':>7s} {'bal-acc':>9s} {'F1':>7s} {'F1 flag-all':>12s} {'specif.':>8s} {'recall':>7s}")
    summ = {}
    for name in res:
        ms = list(out["folds"][name].values())
        summ[name] = {k: float(np.mean([m[k] for m in ms])) for k in ("auc", "bal_acc", "f1", "f1_flag_all", "specificity", "recall")}
        s_ = summ[name]
        print(f"{name:22s} {s_['auc']:7.3f} {s_['bal_acc']:9.3f} {s_['f1']:7.3f} {s_['f1_flag_all']:12.3f} {s_['specificity']:8.2f} {s_['recall']:7.2f}")
    out["mean_over_folds"] = summ

    print("\n" + "=" * 100 + "\nPAIRED DIFFERENCES (mean over folds, 95% bootstrap CI; a CI that excludes 0 = a real difference)\n" + "=" * 100)
    pairs = [("h + size", "size only"), ("h + size", "raw stats + size"), ("h", "size only")]
    ft = [k for k in res if k not in feats]
    pairs += [(f_, "h + size") for f_ in ft] + [(f_, "size only") for f_ in ft]
    for i, A in enumerate(ft):
        for B in ft[i + 1:]:
            pairs.append((A, B))
    out["paired"] = {}
    for A, B in pairs:
        if A in res and B in res:
            for stat in ("auc", "bacc"):
                d, (lo, hi) = paired_diff(res, A, B, stat)
                out["paired"][f"{A} - {B} [{stat}]"] = dict(diff=d, lo=float(lo), hi=float(hi))
                print(f"  {A:20s} minus {B:20s} {stat:5s}: {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {'<-- excludes 0' if lo > 0 or hi < 0 else ''}")
    p = os.path.join(ROOT, "diagnostics", f"folds_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(out, open(p, "w"), indent=2, default=float); print("\nSaved:", p)


if __name__ == "__main__":
    main()