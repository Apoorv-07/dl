r"""
diagnose_scale.py -- READ-ONLY per-feature scale table: graph.pt vs graph_validation.pt.

Put in:  <PROJECT_ROOT>\training\diagnose_scale.py
Run:     python training\diagnose_scale.py *> diag_scale_console.txt
Writes:  <PROJECT_ROOT>\diagnostics\scale_<timestamp>.csv   (new file each run)
"""
import os, sys, time, csv, pickle
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMIT = 500  # sequences per file (benign only for validation) to keep memory small


def collect(path, benign_only):
    seqs = torch.load(path, map_location="cpu", weights_only=False)["sequences"]
    if benign_only:
        seqs = [s for s in seqs if int(s["label"]) == 0]
    seqs = seqs[:LIMIT]
    X = torch.cat([sn.x for s in seqs for sn in s["snapshots"]]).numpy()
    E = torch.cat([sn.edge_attr for s in seqs for sn in s["snapshots"]]).numpy()
    return X, E


def stats(A):
    return dict(min=A.min(0), p50=np.median(A, 0), p99=np.percentile(A, 99, axis=0), max=A.max(0))


def main():
    pk = os.path.join(ROOT, "feature_cols.pkl")
    names = pickle.load(open(pk, "rb")) if os.path.exists(pk) else [f"f{i}" for i in range(53)]
    Xt, Et = collect(os.path.join(ROOT, "graph.pt"), False)
    Xv, Ev = collect(os.path.join(ROOT, "graph_validation.pt"), True)
    rows = []
    for tag, At, Av in (("edge_attr", Et, Ev), ("node_x", Xt, Xv)):
        st, sv = stats(At), stats(Av)
        print(f"\n{tag}: {'idx':>3} {'name':28s} {'train[min,p50,p99,max]':>40s} {'valid[min,p50,p99,max]':>40s}  flag")
        for i in range(At.shape[1]):
            a = [st[k][i] for k in ("min", "p50", "p99", "max")]
            b = [sv[k][i] for k in ("min", "p50", "p99", "max")]
            flag = "MISMATCH" if b[3] > 1.5 * max(a[3], 1e-6) or b[3] > 1.5 else ""
            nm = names[i] if i < len(names) else f"f{i}"
            print(f"{'':10s}{i:>3} {nm:28s} {str([round(float(v), 3) for v in a]):>40s} "
                  f"{str([round(float(v), 3) for v in b]):>40s}  {flag}")
            rows.append([tag, i, nm] + [float(v) for v in a] + [float(v) for v in b] + [flag])
        print(f"{tag}: {sum(1 for r in rows if r[0]==tag and r[-1])} / {At.shape[1]} features flagged")
    os.makedirs(os.path.join(ROOT, "diagnostics"), exist_ok=True)
    out = os.path.join(ROOT, "diagnostics", f"scale_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tensor", "idx", "name"] + [f"train_{k}" for k in ("min", "p50", "p99", "max")]
                   + [f"valid_{k}" for k in ("min", "p50", "p99", "max")] + ["flag"])
        w.writerows(rows)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()