r"""
compare_runs.py -- one comparison table from the eval_*.json files written by eval_exp.py. READ-ONLY, CPU.
Put in:  <PROJECT_ROOT>\training\compare_runs.py
Run:     python training\compare_runs.py *> compare_table.txt
         (optional: --glob "diagnostics\eval_*.json")
Rows: every evaluated model (original checkpoint, untrained control, each experiment) x embedding (h, z) x scorer.
Columns: ROC-AUC, PR-AUC (average precision) with the attack prevalence, AUC on attack windows with >=20% attack flows,
         validation loss and selected epoch (parsed from the run's model string). Scores are never sign-flipped.
"""
import os, re, glob, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--glob", default=os.path.join(ROOT, "diagnostics", "eval_*.json"))
a = ap.parse_args()
rows = []
for p in sorted(glob.glob(a.glob), key=os.path.getmtime):
    r = json.load(open(p)); mm = re.search(r"best epoch (\d+), val loss ([0-9.eE+-]+)", r.get("model", ""))
    ep, vl = (mm.group(1), f"{float(mm.group(2)):.4f}") if mm else ("-", "-")
    for emb in ("h", "z"):
        for sc in ("knn_cos_k5", "mahalanobis"):
            x = r.get(emb, {}).get(sc)
            if x:
                rows.append((r["label"], emb, sc, x["auc"], x["ap"], x["prevalence"], x["auc_attack_ge20pct"], vl, ep))
print(f"{'model':14s} {'emb':3s} {'scorer':12s} {'ROC-AUC':>8s} {'PR-AUC':>7s} {'prev':>6s} {'AUC(>=20%)':>11s} {'val loss':>9s} {'epoch':>6s}")
for r in rows:
    print(f"{r[0]:14s} {r[1]:3s} {r[2]:12s} {r[3]:8.3f} {r[4]:7.3f} {r[5]:6.3f} {r[6]:11.3f} {r[7]:>9s} {r[8]:>6s}")
print("\nPR-AUC must be read against the prevalence column (a random scorer's PR-AUC equals the prevalence).")
print("ROC-AUC below 0.5 = attacks look MORE normal than benign to that scorer; never flipped here.")
print("Validation loss comes from graph.pt held-out benign data and is only comparable between runs with the same batch size and settings.")