r"""
train_exp.py (v2) -- experiment training for the normal-only (benign-only) self-supervised model. Does NOT touch
train_with_validation.py, checkpoints_validation\ or any other run.

Put in:  <PROJECT_ROOT>\training\train_exp.py   (needs exp_common.py next to it; replaces the earlier train_exp.py)

Examples (one change per run):
  python training\train_exp.py --name e1_log  --transform log
  python training\train_exp.py --name e2_pool --pooling meanmax
  python training\train_exp.py --name e3_aug  --edge-drop 0.2 --feature-mask 0.10
  python training\train_exp.py --name e4_reg  --weight-decay 1e-4 --scheduler cosine
  python training\train_exp.py --name e2_pool --pooling meanmax --resume        (continue an interrupted run)
Output:  <PROJECT_ROOT>\<out-root>\<name>\  config.json  best_model.pt  last.pt  training_history.json
         (default out-root = checkpoints_exp; use --out-root checkpoints_improved if you prefer that folder name)
A finished/started run is never overwritten: without --resume it refuses if best_model.pt exists.

Defaults reproduce your original recipe exactly: Adam-style lr 1e-4, batch 32, tau 0.1, no weight decay, no scheduler, edge drop 0.5,
feature mask 0.15, 90/10 split seed 42, early stopping patience 10 / min delta 1e-4, max 100 epochs.
Model selection uses ONLY the benign held-out contrastive loss of graph.pt; no attack labels are ever used.

Per-epoch diagnostics (validation, augmented views): loss, learning rate, positive-pair cosine, mean negative-pair cosine,
embedding spread (mean per-dimension std of normalised z; near 0 = collapse), raw z norm.
"""
import os, sys, json, time, argparse, platform
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from exp_common import build_models, modules, set_mode, encode_sequence, DEV
from training.augmentations import create_two_views
from losses.nt_xent import NTXentLoss

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--out-root", default="checkpoints_exp")
ap.add_argument("--transform", choices=["none", "log"], default="none")
ap.add_argument("--pooling", choices=["mean", "meanmax"], default="mean")
ap.add_argument("--edge-drop", type=float, default=0.5)
ap.add_argument("--feature-mask", type=float, default=0.15)
ap.add_argument("--temperature", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--weight-decay", type=float, default=0.0)
ap.add_argument("--scheduler", choices=["none", "cosine"], default="none")
ap.add_argument("--max-epochs", type=int, default=100)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

MIN_DELTA, SPLIT = 1e-4, 0.90
cfg = dict(transform=a.transform, pooling=a.pooling, edge_drop=a.edge_drop, feature_mask=a.feature_mask)
out_dir = os.path.join(ROOT, a.out_root, a.name)
best_path, last_path = os.path.join(out_dir, "best_model.pt"), os.path.join(out_dir, "last.pt")
hist_path = os.path.join(out_dir, "training_history.json")
if os.path.exists(best_path) and not a.resume:
    sys.exit(f"STOP: {best_path} already exists. Use a new --name, or --resume to continue it (nothing is overwritten).")
if a.resume and not os.path.exists(last_path):
    sys.exit(f"STOP: nothing to resume ({last_path} not found)")
os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(a.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(a.seed)

seqs = torch.load(os.path.join(ROOT, "graph.pt"), weights_only=False)["sequences"]
perm = torch.randperm(len(seqs), generator=torch.Generator().manual_seed(a.seed)).tolist()
cut = int(SPLIT * len(seqs))
train_seqs = [seqs[i] for i in perm[:cut]]; val_seqs = [seqs[i] for i in perm[cut:]]
del seqs
print(f"run: {a.name}   cfg: {cfg}\ndevice: {DEV}  train {len(train_seqs)}  val {len(val_seqs)}  seed {a.seed}")

m = build_models(cfg)
params = [p for mod in modules(m) for p in mod.parameters()]
print("trainable parameters:", sum(p.numel() for p in params))
opt = (AdamW if a.weight_decay > 0 else Adam)(params, lr=a.lr, **({"weight_decay": a.weight_decay} if a.weight_decay > 0 else {}))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.max_epochs) if a.scheduler == "cosine" else None
crit = NTXentLoss(temperature=a.temperature)

if not a.resume:
    json.dump(dict(args=vars(a), cfg=cfg, torch=torch.__version__, python=platform.python_version(),
                   train_sequences=len(train_seqs), val_sequences=len(val_seqs)),
              open(os.path.join(out_dir, "config.json"), "w"), indent=2)


def forward_batch(batch):
    z1, z2 = [], []
    for item in batch:
        v1, v2 = create_two_views(item["snapshots"], a.edge_drop, a.feature_mask)
        z1.append(encode_sequence(m, v1, cfg)[1]); z2.append(encode_sequence(m, v2, cfg)[1])
    return torch.cat(z1, 0), torch.cat(z2, 0)


@torch.no_grad()
def diag(z1, z2):
    n1, n2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    sim = n1 @ n2.T; eye = torch.eye(len(sim), dtype=torch.bool, device=sim.device)
    return dict(pos_cos=float(sim.diag().mean()), neg_cos=float(sim[~eye].mean()),
                spread=float(n1.std(0).mean()), z_norm=float(z1.norm(dim=1).mean()))


def save(path, epoch, tl, vl, best_vl, best_ep, pat):
    torch.save(dict(epoch=epoch, cfg=cfg, args=vars(a),
                    gat_state_dict=m["gat"].state_dict(), pool_state_dict=m["pool"].state_dict(),
                    transformer_state_dict=m["tr"].state_dict(), projection_state_dict=m["pr"].state_dict(),
                    optimizer_state_dict=opt.state_dict(), scheduler_state_dict=sched.state_dict() if sched else None,
                    train_loss=tl, validation_loss=vl, best_validation_loss=best_vl, best_epoch=best_ep, patience_counter=pat,
                    batch_size=a.batch_size, learning_rate=a.lr, nt_xent_temperature=a.temperature, seed=a.seed), path)


best_vl, best_ep, pat, hist, start = float("inf"), None, 0, [], 1
if a.resume:
    ck = torch.load(last_path, map_location=DEV, weights_only=False)
    m["gat"].load_state_dict(ck["gat_state_dict"]); m["pool"].load_state_dict(ck["pool_state_dict"])
    m["tr"].load_state_dict(ck["transformer_state_dict"]); m["pr"].load_state_dict(ck["projection_state_dict"])
    opt.load_state_dict(ck["optimizer_state_dict"])
    if sched and ck.get("scheduler_state_dict"):
        sched.load_state_dict(ck["scheduler_state_dict"])
    best_vl, best_ep, pat = ck["best_validation_loss"], ck["best_epoch"], ck["patience_counter"]
    hist = json.load(open(hist_path))["epochs"][: ck["epoch"]]
    start = ck["epoch"] + 1
    print(f"RESUMED from epoch {ck['epoch']} (best {best_vl:.6f} @ {best_ep}, patience {pat}); data order/augmentation RNG is not restored")

for epoch in range(start, a.max_epochs + 1):
    t0 = time.time(); set_mode(m, True)
    order = torch.randperm(len(train_seqs)).tolist(); nb = len(order) // a.batch_size; tl = 0.0
    for b in range(nb):
        batch = [train_seqs[i] for i in order[b * a.batch_size:(b + 1) * a.batch_size]]
        loss = crit(*forward_batch(batch))
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
    tl /= nb
    set_mode(m, False); vl = 0.0; nv = len(val_seqs) // a.batch_size; d_acc = []
    with torch.no_grad():
        for b in range(nv):
            z1, z2 = forward_batch(val_seqs[b * a.batch_size:(b + 1) * a.batch_size])
            vl += crit(z1, z2).item(); d_acc.append(diag(z1, z2))
    vl /= nv; d = {k: sum(x[k] for x in d_acc) / len(d_acc) for k in d_acc[0]}
    lr_now = opt.param_groups[0]["lr"]
    if sched:
        sched.step()
    is_best = vl < best_vl - MIN_DELTA
    if is_best:
        best_vl, best_ep, pat = vl, epoch, 0
        save(best_path, epoch, tl, vl, best_vl, best_ep, pat)
    else:
        pat += 1
    save(last_path, epoch, tl, vl, best_vl, best_ep, pat)
    hist.append(dict(epoch=epoch, train_loss=tl, validation_loss=vl, best_validation_loss=best_vl, best_epoch=best_ep,
                     patience_counter=pat, lr=lr_now, **d))
    json.dump(dict(cfg=cfg, args=vars(a), epochs=hist, best_epoch=best_ep, best_validation_loss=best_vl), open(hist_path, "w"), indent=2)
    print(f"epoch {epoch:3d} | train {tl:.5f} | val {vl:.5f} | best {best_vl:.5f} @ {best_ep} | pat {pat}/{a.patience} | lr {lr_now:.2e} | "
          f"pos {d['pos_cos']:.3f} neg {d['neg_cos']:.3f} spread {d['spread']:.3f} | {time.time() - t0:.0f}s" + ("  NEW BEST" if is_best else ""), flush=True)
    if pat >= a.patience:
        print("early stopping."); break
print(f"\nDONE. best epoch {best_ep}, best val loss {best_vl:.6f}\nbest checkpoint: {best_path}")