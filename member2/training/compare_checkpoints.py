r"""
compare_checkpoints.py -- READ-ONLY apples-to-apples comparison of training checkpoints
on the held-out 10% of graph.pt (the same split the training script used, seed 42).

Put in:  <PROJECT_ROOT>\training\compare_checkpoints.py
Run:
  python training\compare_checkpoints.py `
      --ckpt bs32=checkpoints_validation\best_model.pt `
      --ckpt bs4=<path to your batch-4 checkpoint>.pt `
      *> compare_console.txt
Writes: <PROJECT_ROOT>\diagnostics\compare_<timestamp>.json (new file each run).
Never modifies any checkpoint or dataset.

Every checkpoint sees the SAME augmented views (generated once, fixed seed), so
differences come from the checkpoints, not from random augmentation.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from models.gatv2_encoder import GATv2Encoder
from models.temporal_transformer import TemporalTransformer
from models.projection_head import ProjectionHead
from models.global_pooling import global_mean_pooling
from losses.nt_xent import NTXentLoss
from training.augmentations import create_two_views

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def load_ckpt(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    gat = GATv2Encoder(53, 128, 4, 0.2).to(DEV)
    tr = TemporalTransformer(128, 4, 2, 0.1).to(DEV)
    pr = ProjectionHead(128, 128, 64).to(DEV)
    gat.load_state_dict(ck["gat_state_dict"])          # strict: raises if architecture differs
    tr.load_state_dict(ck["transformer_state_dict"])
    pr.load_state_dict(ck["projection_state_dict"])
    for m in (gat, tr, pr):
        m.eval()
    meta = {k: ck.get(k) for k in ("epoch", "best_epoch", "best_validation_loss", "batch_size",
                                   "learning_rate", "nt_xent_temperature",
                                   "augmentation_dropout_probability",
                                   "augmentation_feature_mask_probability")}
    return gat, tr, pr, meta


@torch.no_grad()
def encode(models, seq):
    gat, tr, pr = models
    embs = []
    for g in seq:
        x = g.x.to(DEV).clamp(-5, 5)
        ea = g.edge_attr.to(DEV).clamp(-5, 5)
        embs.append(global_mean_pooling(gat(x, g.edge_index.to(DEV), ea)))
    h = tr(torch.cat(embs, 0).unsqueeze(0))
    return h.squeeze(0), pr(h).squeeze(0)


def pair_cos(E, n=500):
    En = F.normalize(E, dim=1); idx = torch.randperm(len(En), generator=torch.Generator().manual_seed(0))[:n]
    S = En[idx] @ En[idx].T
    return float(S[~torch.eye(len(idx), dtype=torch.bool, device=S.device)].mean())


def retrieval_top1(A, B):
    """For each i, is B[i] the nearest of all B to A[i]? Independent of batch size."""
    S = F.normalize(A, dim=1) @ F.normalize(B, dim=1).T
    return float((S.argmax(1) == torch.arange(len(A), device=S.device)).float().mean())


def batched_loss(z1, z2, bs, temp):
    crit = NTXentLoss(temperature=temp)
    n = (len(z1) // bs) * bs
    losses = [float(crit(z1[i:i + bs], z2[i:i + bs])) for i in range(0, n, bs)]
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=path")
    ap.add_argument("--temperature", type=float, default=0.1)
    a = ap.parse_args()

    data = torch.load(os.path.join(ROOT, "graph.pt"), map_location="cpu", weights_only=False)["sequences"]
    perm = torch.randperm(len(data), generator=torch.Generator().manual_seed(SEED)).tolist()
    val = [data[i] for i in perm[int(0.9 * len(data)):]]
    print(f"held-out graph.pt sequences: {len(val)}")

    torch.manual_seed(SEED)                       # fixed views shared by all checkpoints
    views = [create_two_views(s["snapshots"], 0.5, 0.15) for s in val]

    out = {"n_val": len(val), "chance_top1": 1.0 / len(val), "runs": {}}
    for item in a.ckpt:
        label, path = item.split("=", 1)
        gat, tr, pr, meta = load_ckpt(path)
        m = (gat, tr, pr)
        H1, Z1, H2, Z2, Hc = [], [], [], [], []
        for (v1, v2), s in zip(views, val):
            h1, z1 = encode(m, v1); h2, z2 = encode(m, v2); hc, _ = encode(m, s["snapshots"])
            H1.append(h1); Z1.append(z1); H2.append(h2); Z2.append(z2); Hc.append(hc)
        H1, Z1, H2, Z2, Hc = (torch.stack(t) for t in (H1, Z1, H2, Z2, Hc))
        r = dict(meta=meta,
                 ntxent_bs32=batched_loss(Z1, Z2, 32, a.temperature),
                 ntxent_bs4=batched_loss(Z1, Z2, 4, a.temperature),
                 chance_bs32=float(np.log(63)), chance_bs4=float(np.log(7)),
                 retrieval_top1_z=retrieval_top1(Z1, Z2),
                 retrieval_top1_h=retrieval_top1(H1, H2),
                 pairwise_cos_h_clean=pair_cos(Hc),
                 pairwise_cos_z_aug=pair_cos(Z1))
        out["runs"][label] = r
        print(f"\n[{label}] {path}")
        for k, v in r.items():
            print(f"  {k}: {v}")
    print(f"\nrandom-guess top-1 retrieval = {out['chance_top1']:.4f}")

    os.makedirs(os.path.join(ROOT, "diagnostics"), exist_ok=True)
    p = os.path.join(ROOT, "diagnostics", f"compare_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(out, open(p, "w"), indent=2, default=float)
    print("Saved:", p)


if __name__ == "__main__":
    main()