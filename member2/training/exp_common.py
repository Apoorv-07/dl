r"""
exp_common.py -- shared helpers for the experiment scripts (train_exp.py, eval_exp.py).
Put in: <PROJECT_ROOT>\training\exp_common.py

It does NOT change any existing file. It only adds two OPTIONAL switches on top of your current model:
  transform : "none" (current behaviour) | "log"   -> v -> log1p(A*v)/log1p(A), A=100, applied to node and edge features
                                                      AFTER augmentation, identically in training and evaluation
  pooling   : "mean" (current behaviour) | "meanmax" -> concat(mean, max over nodes) -> Linear(256,128)
With transform="none" and pooling="mean" the model is exactly your current architecture.
"""
import os, sys
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from models.gatv2_encoder import GATv2Encoder
from models.temporal_transformer import TemporalTransformer
from models.projection_head import ProjectionHead
from models.global_pooling import global_mean_pooling

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_A = 100.0


class Pool(nn.Module):
    """mean pooling (parameter-free, identical to the current one) or mean+max with a learned linear map."""

    def __init__(self, mode="mean", d=128):
        super().__init__()
        self.mode = mode
        if mode == "meanmax":
            self.proj = nn.Linear(2 * d, d)

    def forward(self, node_emb):
        mean = global_mean_pooling(node_emb)                 # [1, d]  (same function as before)
        if self.mode == "mean":
            return mean
        mx = node_emb.max(dim=0, keepdim=True).values        # [1, d]
        return self.proj(torch.cat([mean, mx], dim=1))


def default_cfg():
    return dict(transform="none", pooling="mean", edge_drop=0.5, feature_mask=0.15)


def build_models(cfg):
    gat = GATv2Encoder(53, 128, 4, 0.2).to(DEV)
    pool = Pool(cfg["pooling"]).to(DEV)
    tr = TemporalTransformer(128, 4, 2, 0.1).to(DEV)
    pr = ProjectionHead(128, 128, 64).to(DEV)
    return dict(gat=gat, pool=pool, tr=tr, pr=pr)


def modules(m):
    return [m["gat"], m["pool"], m["tr"], m["pr"]]


def set_mode(m, train):
    for x in modules(m):
        x.train(train)


def _prep(v, transform):
    v = v.to(DEV).clamp(-5.0, 5.0)                           # same clamp as training
    if transform == "log":
        v = torch.log1p(LOG_A * v.clamp(min=0.0)) / torch.log1p(torch.tensor(LOG_A, device=v.device))
    return v


def encode_sequence(m, snapshots, cfg):
    toks = []
    for g in snapshots:
        x = _prep(g.x, cfg["transform"]); ea = _prep(g.edge_attr, cfg["transform"])
        toks.append(m["pool"](m["gat"](x, g.edge_index.to(DEV), ea)))
    h = m["tr"](torch.cat(toks, 0).unsqueeze(0))
    return h, m["pr"](h)


@torch.no_grad()
def embed(m, seqs, cfg):
    set_mode(m, False)
    H, Z = [], []
    for s in seqs:
        h, z = encode_sequence(m, s["snapshots"], cfg)
        H.append(h.squeeze(0).cpu()); Z.append(z.squeeze(0).cpu())
    return torch.stack(H).numpy(), torch.stack(Z).numpy()


def load_checkpoint(path):
    """Works for your existing checkpoints (no cfg -> transform none, pooling mean) and for new experiment ones."""
    ck = torch.load(path, map_location=DEV, weights_only=False)
    cfg = ck.get("cfg", default_cfg())
    m = build_models(cfg)
    m["gat"].load_state_dict(ck["gat_state_dict"])
    m["tr"].load_state_dict(ck["transformer_state_dict"])
    m["pr"].load_state_dict(ck["projection_state_dict"])
    if "pool_state_dict" in ck:
        m["pool"].load_state_dict(ck["pool_state_dict"])
    set_mode(m, False)
    return m, cfg, ck