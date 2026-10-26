r"""
train_exp.py -- NEW experiment training script. Does NOT touch train_with_validation.py or checkpoints_validation\.

Put in:  <PROJECT_ROOT>\training\train_exp.py   (needs exp_common.py next to it)
Run one experiment at a time, for example:
  python training\train_exp.py --name e1_log  --transform log
  python training\train_exp.py --name e2_pool --pooling meanmax
  python training\train_exp.py --name e3_aug  --edge-drop 0.2 --feature-mask 0.10
Saves to:  <PROJECT_ROOT>\checkpoints_exp\<name>\best_model.pt + training_history.json
Refuses to run if that folder already has a best_model.pt (no overwriting).

Everything else is identical to your current training: same graph.pt, same 90/10 split (seed 42),
Adam lr 1e-4, batch 32, NT-Xent tau 0.1, early stopping patience 10 / min_delta 1e-4, max 100 epochs.
Model selection uses only the benign held-out loss of graph.pt. No attack labels are used.
"""
import os, sys, json, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
import torch
from torch.optim import Adam
from exp_common import build_models, modules, set_mode, encode_sequence, DEV
from training.augmentations import create_two_views
from losses.nt_xent import NTXentLoss

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--transform", choices=["none", "log"], default="none")
ap.add_argument("--pooling", choices=["mean", "meanmax"], default="mean")
ap.add_argument("--edge-drop", type=float, default=0.5)
ap.add_argument("--feature-mask", type=float, default=0.15)
ap.add_argument("--max-epochs", type=int, default=100)
ap.add_argument("--batch-size", type=int, default=32)
a = ap.parse_args()

SEED, LR, TAU, PATIENCE, MIN_DELTA, SPLIT = 42, 1e-4, 0.1, 10, 1e-4, 0.90
cfg = dict(transform=a.transform, pooling=a.pooling, edge_drop=a.edge_drop, feature_mask=a.feature_mask)
out_dir = os.path.join(ROOT, "checkpoints_exp", a.name)
best_path = os.path.join(out_dir, "best_model.pt")
if os.path.exists(best_path):
    sys.exit(f"STOP: {best_path} already exists. Choose a new --name (nothing is overwritten).")
os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

seqs = torch.load(os.path.join(ROOT, "graph.pt"), weights_only=False)["sequences"]
perm = torch.randperm(len(seqs), generator=torch.Generator().manual_seed(SEED)).tolist()
cut = int(SPLIT * len(seqs))
train_seqs = [seqs[i] for i in perm[:cut]]; val_seqs = [seqs[i] for i in perm[cut:]]
print(f"experiment: {a.name}  cfg: {cfg}\ndevice: {DEV}  train {len(train_seqs)}  val {len(val_seqs)}")

m = build_models(cfg)
params = [p for mod in modules(m) for p in mod.parameters()]
print("trainable parameters:", sum(p.numel() for p in params))
opt = Adam(params, lr=LR)
crit = NTXentLoss(temperature=TAU)


def batch_loss(batch):
    z1, z2 = [], []
    for item in batch:
        v1, v2 = create_two_views(item["snapshots"], a.edge_drop, a.feature_mask)
        z1.append(encode_sequence(m, v1, cfg)[1]); z2.append(encode_sequence(m, v2, cfg)[1])
    return crit(torch.cat(z1, 0), torch.cat(z2, 0))


def save(path, epoch, tl, vl, best_vl, best_ep, pat):
    torch.save(dict(epoch=epoch, cfg=cfg,
                    gat_state_dict=m["gat"].state_dict(), pool_state_dict=m["pool"].state_dict(),
                    transformer_state_dict=m["tr"].state_dict(), projection_state_dict=m["pr"].state_dict(),
                    optimizer_state_dict=opt.state_dict(), train_loss=tl, validation_loss=vl,
                    best_validation_loss=best_vl, best_epoch=best_ep, patience_counter=pat,
                    batch_size=a.batch_size, learning_rate=LR, nt_xent_temperature=TAU, seed=SEED), path)


best_vl, best_ep, pat, hist = float("inf"), None, 0, []
for epoch in range(1, a.max_epochs + 1):
    t0 = time.time(); set_mode(m, True)
    order = torch.randperm(len(train_seqs)).tolist(); nb = len(order) // a.batch_size; tl = 0.0
    for b in range(nb):
        batch = [train_seqs[i] for i in order[b * a.batch_size:(b + 1) * a.batch_size]]
        loss = batch_loss(batch)
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
    tl /= nb
    set_mode(m, False); vl = 0.0; nv = len(val_seqs) // a.batch_size
    with torch.no_grad():
        for b in range(nv):
            vl += batch_loss(val_seqs[b * a.batch_size:(b + 1) * a.batch_size]).item()
    vl /= nv
    is_best = vl < best_vl - MIN_DELTA
    if is_best:
        best_vl, best_ep, pat = vl, epoch, 0
        save(best_path, epoch, tl, vl, best_vl, best_ep, pat)
    else:
        pat += 1
    hist.append(dict(epoch=epoch, train_loss=tl, validation_loss=vl, best_validation_loss=best_vl,
                     best_epoch=best_ep, patience_counter=pat))
    json.dump(dict(cfg=cfg, epochs=hist, best_epoch=best_ep, best_validation_loss=best_vl),
              open(os.path.join(out_dir, "training_history.json"), "w"), indent=2)
    print(f"epoch {epoch:3d} | train {tl:.5f} | val {vl:.5f} | best {best_vl:.5f} @ {best_ep} | "
          f"patience {pat}/{PATIENCE} | {time.time() - t0:.0f}s" + ("  NEW BEST" if is_best else ""), flush=True)
    if pat >= PATIENCE:
        print("early stopping."); break
print(f"\nDONE. best epoch {best_ep}, best val loss {best_vl:.6f}\ncheckpoint: {best_path}")