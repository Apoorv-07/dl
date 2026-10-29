r"""
model_adapter.py -- inference adapter for the Member 2 GATv2 + Temporal Transformer encoder (frozen checkpoint).
Put in:  <PROJECT_ROOT>\training\model_adapter.py   (needs exp_common.py in the same folder)

    from model_adapter import Member2Adapter
    ad = Member2Adapter("checkpoints_validation/best_model.pt")
    h = ad.encode(sequence["snapshots"])                  # np.ndarray (128,), unit L2 norm  <- THE embedding to use
    out = ad.encode_with_attention(sequence["snapshots"]) # h, z + GAT edge attention + temporal attention
    ad.metadata()                                         # dict: architecture, checkpoint info, input contract, versions

INPUT CONTRACT (must hold or the embedding is meaningless)
  * a sequence = list of exactly 5 torch_geometric Data snapshots (x [N,53], edge_index [2,E], edge_attr [E,53])
  * features already on graph.pt scale (all in [0,1]); this adapter does NOT rescale, it only clamps to [-5,5]
  * evaluation mode, no augmentation, no gradient
`h` (128-d, L2-normalised) is the intended representation. `z` (64-d) is the contrastive projection and is only
returned for completeness; do not use it for scoring.

Self-test (no retraining, checks that the attention path reproduces the normal path):
  python training\model_adapter.py --ckpt checkpoints_validation\best_model.pt --graph graph_validation_v2.pt --n 3
"""
import os, sys, platform, argparse
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from exp_common import load_checkpoint, encode_sequence, _prep, DEV


class Member2Adapter:
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.model, self.cfg, self.ckpt = load_checkpoint(ckpt_path)   # strict load, eval mode

    # ------------------------------------------------------------------ plain embedding
    @torch.no_grad()
    def encode(self, snapshots):
        h, _ = encode_sequence(self.model, snapshots, self.cfg)
        return h.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_sequences(self, sequences):
        return np.stack([self.encode(s["snapshots"]) for s in sequences])

    # ------------------------------------------------------------------ embedding + attention
    @torch.no_grad()
    def encode_with_attention(self, snapshots):
        """
        Returns dict:
          h, z                 : np arrays (128,), (64,)
          gat_attention        : list over the 5 snapshots of
                                   {"layer1": (edge_index [2,E'], alpha [E',4]),   # 4 heads
                                    "layer2": (edge_index [2,E'], alpha [E',1])}
                                 NOTE: PyG adds self-loops, so E' = E + N and the last N edges are the self-loops.
          temporal_attention   : list over the 2 transformer layers of np arrays [heads=4, 5, 5]
                                 (row = query snapshot, column = key snapshot)
        """
        m, cfg = self.model, self.cfg
        gat, tr = m["gat"], m["tr"]
        toks, gat_att = [], []
        for g in snapshots:
            x = _prep(g.x, cfg["transform"]); ea = _prep(g.edge_attr, cfg["transform"]); ei = g.edge_index.to(DEV)
            x1, (ei1, a1) = gat.gat1(x, ei, ea, return_attention_weights=True)
            x1 = gat.activation(x1)
            x2, (ei2, a2) = gat.gat2(x1, ei, ea, return_attention_weights=True)
            toks.append(m["pool"](x2))
            gat_att.append({"layer1": (ei1.cpu().numpy(), a1.cpu().numpy()),
                            "layer2": (ei2.cpu().numpy(), a2.cpu().numpy())})
        t = torch.cat(toks, 0).unsqueeze(0)                                  # [1,5,128]
        x = t + tr.pos_embedding[:, : t.size(1), :]
        temporal = []
        for layer in tr.transformer.layers:
            assert not layer.norm_first, "adapter assumes post-LN (the trained configuration)"
            sa, w = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
            x = layer.norm1(x + layer.dropout1(sa))
            x = layer.norm2(x + layer._ff_block(x))
            temporal.append(w.squeeze(0).cpu().numpy())
        h = F.normalize(tr.norm(x.mean(dim=1)), p=2, dim=-1)
        z = m["pr"](h)
        return dict(h=h.squeeze(0).cpu().numpy().astype(np.float32), z=z.squeeze(0).cpu().numpy().astype(np.float32),
                    gat_attention=gat_att, temporal_attention=temporal)

    # ------------------------------------------------------------------ metadata
    def metadata(self):
        import torch_geometric
        ck = self.ckpt
        return dict(
            model=("GATv2 (2 layers) -> " +
                   ("global mean pooling per snapshot" if self.cfg["pooling"] == "mean"
                    else "concat(mean, max over nodes) -> Linear(256,128) per snapshot") +
                   " -> Temporal Transformer -> h -> projection head -> z"),
            embedding_used_downstream="h", h_dim=128, h_l2_normalised=True, z_dim=64,
            gatv2=dict(layer1="GATv2Conv(53 -> 4 heads x 128, concat=512, edge_dim=53, dropout=0.2) + ELU",
                       layer2="GATv2Conv(512 -> 128, 1 head, edge_dim=53, dropout=0.2)",
                       pyg_defaults="add_self_loops=True, negative_slope=0.2"),
            pooling=self.cfg["pooling"],
            temporal=dict(snapshots_per_sequence=5, positional_embedding="learned (1,5,128)", layers=2, heads=4,
                          ffn=512, activation="gelu", dropout=0.1, norm="post-LN",
                          readout="mean over 5 tokens -> LayerNorm -> L2 normalise"),
            projection_head="Linear(128,128) -> GELU -> Linear(128,64)",
            input=dict(node_features=53, edge_features=53, clamp=[-5.0, 5.0], input_transform=self.cfg["transform"],
                       expected_scale="values in [0,1] (graph.pt scale); NOT rescaled by the model"),
            checkpoint=dict(path=self.ckpt_path, epoch=ck.get("epoch"), best_epoch=ck.get("best_epoch"),
                            best_validation_loss=ck.get("best_validation_loss"), batch_size=ck.get("batch_size"),
                            learning_rate=ck.get("learning_rate"), nt_xent_temperature=ck.get("nt_xent_temperature"),
                            seed=ck.get("seed"),
                            num_parameters=int(sum(p.numel() for mod in (self.model["gat"], self.model["pool"],
                                                                         self.model["tr"], self.model["pr"])
                                                   for p in mod.parameters()))),
            versions=dict(python=platform.python_version(), torch=torch.__version__,
                          torch_geometric=torch_geometric.__version__, numpy=np.__version__),
        )


def _selftest(a):
    ad = Member2Adapter(a.ckpt)
    seqs = torch.load(a.graph, map_location="cpu", weights_only=False)["sequences"][: a.n]
    worst = 0.0
    for i, s in enumerate(seqs):
        h = ad.encode(s["snapshots"]); o = ad.encode_with_attention(s["snapshots"])
        d = float(np.abs(h - o["h"]).max()); worst = max(worst, d)
        print(f"seq {i}: |h|={np.linalg.norm(h):.6f}  max|h_plain - h_attention_path|={d:.2e}  "
              f"temporal attn shapes={[w.shape for w in o['temporal_attention']]}  "
              f"gat layer1 alpha shape snapshot0={o['gat_attention'][0]['layer1'][1].shape}")
        rows = o["temporal_attention"][0].sum(-1)
        assert np.allclose(rows, 1.0, atol=1e-4), "temporal attention rows must sum to 1"
    print("\nPASS" if worst < 1e-3 else "\nFAIL: attention path does not reproduce the normal path", f"(worst diff {worst:.2e})")
    print("metadata:", ad.metadata())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=3)
    _selftest(ap.parse_args())