r"""
finetune_sup.py -- SUPERVISED fine-tuning of the encoder + a small head, one chronological fold at a time (GPU).
Never touches your existing checkpoint: it loads it, trains a COPY, and writes only to diagnostics\.

Put in:  <PROJECT_ROOT>\training\finetune_sup.py   (needs exp_common.py and sup_common.py next to it)
Run one fold at a time (folds 2,3,4,5), for the pretrained start and for the from-scratch control:
  python training\finetune_sup.py --records member3_package_20261028_033449\member2_records.csv --val graph_validation_v2.pt --ckpt checkpoints_validation\best_model.pt --fold 2
  python training\finetune_sup.py --records member3_package_20261028_033449\member2_records.csv --val graph_validation_v2.pt --fold 2 --random-init
Writes:  diagnostics\ft_<label>_fold<k>_<timestamp>.npz and .json (new files each run)
Then evaluate everything together with sup_folds.py --ft "diagnostics\ft_*.npz".

Method: input = [h * sqrt(128), standardised log edge count] -> Linear -> attack logit. Loss: BCE with pos_weight.
Encoder lr 2e-5, head lr 3e-3, AdamW, batch of 8 windows (gradient accumulation, one window at a time), max 12 epochs,
early stopping on CALIB AUC (patience 3). The threshold is chosen later on CALIB only. TEST is never used for any choice.
SUPERVISED: uses attack labels. Report it as supervised, in-domain, rolling-origin chronological evaluation.
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.append(HERE); sys.path.append(ROOT)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from exp_common import load_checkpoint, build_models, default_cfg, modules, set_mode, encode_sequence, DEV
from sup_common import load_rows, make_folds, youden, metrics

ap = argparse.ArgumentParser()
ap.add_argument("--records", required=True); ap.add_argument("--val", required=True)
ap.add_argument("--ckpt", default=None); ap.add_argument("--random-init", action="store_true")
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--enc-lr", type=float, default=2e-5); ap.add_argument("--head-lr", type=float, default=3e-3)
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()
if not a.random_init and not a.ckpt:
    sys.exit("give --ckpt <path> (pretrained start) or --random-init (from-scratch control)")
label = "supervised from scratch" if a.random_init else "fine-tuned (pretrained)"

D = load_rows(a.records); y = D["y"]
fold = next((f for f in make_folds(D) if f["k"] == a.fold), None)
if fold is None or not fold["ok"]:
    sys.exit(f"fold {a.fold} is not usable (valid folds are 2..5 and each block needs both classes)")
fit, cal, te = fold["fit"], fold["cal"], fold["test"]
print(f"{label} | fold {a.fold}: fit {len(fit)}  calib {len(cal)}  test {len(te)}  device {DEV}")
vas = torch.load(a.val, map_location="cpu", weights_only=False)["sequences"]

torch.manual_seed(42 + a.fold); rng = np.random.default_rng(42 + a.fold)
if a.random_init:
    cfg = default_cfg(); m = build_models(cfg)
else:
    m, cfg, _ = load_checkpoint(a.ckpt)
head = nn.Linear(129, 1).to(DEV)
mu, sd = D["size"][fit].mean(), D["size"][fit].std() + 1e-8
enc_params = [p for mod in (m["gat"], m["pool"], m["tr"]) for p in mod.parameters()]
opt = torch.optim.AdamW([dict(params=enc_params, lr=a.enc_lr), dict(params=head.parameters(), lr=a.head_lr)], weight_decay=1e-4)
n_pos = int(y[fit].sum()); pos_weight = torch.tensor((len(fit) - n_pos) / max(n_pos, 1), device=DEV)


def logit(i):
    seq = vas[int(D["src"][i])]
    h, _ = encode_sequence(m, seq["snapshots"], cfg)                       # [1,128], unit norm
    s = torch.tensor([[(D["size"][i, 0] - mu) / sd]], dtype=torch.float32, device=DEV)
    return head(torch.cat([h * (128 ** 0.5), s], 1)).squeeze()


@torch.no_grad()
def scores(idx):
    set_mode(m, False); head.eval()
    return np.array([logit(i).item() for i in idx])


def snapshot():
    return [{k: v.clone() for k, v in mod.state_dict().items()} for mod in (m["gat"], m["pool"], m["tr"], head)]


def restore(st):
    for mod, s in zip((m["gat"], m["pool"], m["tr"], head), st):
        mod.load_state_dict(s)


best_auc, best_state, bad, hist = -1.0, snapshot(), 0, []
auc0 = roc_auc_score(y[cal], scores(cal)) if not a.random_init else float("nan")
print(f"calib AUC before training: {auc0:.4f}" if not a.random_init else "from scratch: no pre-training baseline")
for ep in range(1, a.epochs + 1):
    t0 = time.time(); set_mode(m, True); head.train(); opt.zero_grad()
    order = rng.permutation(fit); tl = 0.0; cnt = 0
    for j, i in enumerate(order, 1):
        yi = torch.tensor(float(y[i]), device=DEV)
        loss = F.binary_cross_entropy_with_logits(logit(i), yi, pos_weight=pos_weight if y[i] == 1 else None)
        (loss / a.batch).backward(); tl += loss.item(); cnt += 1
        if j % a.batch == 0:
            torch.nn.utils.clip_grad_norm_(enc_params + list(head.parameters()), 1.0); opt.step(); opt.zero_grad()
    opt.zero_grad()
    auc = float(roc_auc_score(y[cal], scores(cal)))
    hist.append(dict(epoch=ep, train_loss=tl / cnt, calib_auc=auc))
    improved = auc > best_auc + 1e-4
    if improved:
        best_auc, best_state, bad = auc, snapshot(), 0
    else:
        bad += 1
    print(f"epoch {ep:2d} | train loss {tl / cnt:.4f} | calib AUC {auc:.4f} | best {best_auc:.4f} | patience {bad}/{a.patience} | "
          f"{time.time() - t0:.0f}s" + ("  NEW BEST" if improved else ""), flush=True)
    if bad >= a.patience:
        print("early stopping."); break
restore(best_state)
s_cal, s_te = scores(cal), scores(te)
thr = youden(y[cal], s_cal); res = metrics(y[te], s_te, thr)
print(f"\nTEST (fold {a.fold}, threshold from CALIB): AUC={res['auc']:.3f} bal-acc={res['bal_acc']:.3f} F1={res['f1']:.3f} "
      f"(flag-all {res['f1_flag_all']:.3f}) specificity={res['specificity']:.2f} recall={res['recall']:.2f}")
os.makedirs(os.path.join(ROOT, "diagnostics"), exist_ok=True)
tag = f"{'scratch' if a.random_init else 'pretrained'}_fold{a.fold}_{time.strftime('%Y%m%d_%H%M%S')}"
np.savez(os.path.join(ROOT, "diagnostics", f"ft_{tag}.npz"), label=label, fold=a.fold, test_src=D["src"][te], test_scores=s_te,
         calib_y=y[cal], calib_scores=s_cal)
json.dump(dict(label=label, fold=a.fold, args=vars(a), history=hist, calib_auc_best=best_auc, test=res),
          open(os.path.join(ROOT, "diagnostics", f"ft_{tag}.json"), "w"), indent=2)
print("Saved:", os.path.join(ROOT, "diagnostics", f"ft_{tag}.npz"))