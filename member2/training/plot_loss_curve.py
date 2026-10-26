import json
import os
import sys
import matplotlib.pyplot as plt

# Project Root Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

HISTORY_PATH = os.path.join(
    PROJECT_ROOT, "checkpoints_validation", "training_history.json"
)

if not os.path.exists(HISTORY_PATH):
  print(f"Error: {HISTORY_PATH} not found.")
  sys.exit(1)

with open(HISTORY_PATH, "r") as f:
  data = json.load(f)

epochs = [e["epoch"] for e in data["epochs"]]
train_loss = [e["train_loss"] for e in data["epochs"]]
val_loss = [e["validation_loss"] for e in data["epochs"]]
best_epoch = data["best_epoch"]
best_val_loss = data["best_validation_loss"]

print("=" * 60)
print("PRE-TRAINING LOSS CONVERGENCE SUMMARY")
print("=" * 60)
print(f"Total Epochs Run      : {len(epochs)}")
print(f"Best Epoch            : Epoch {best_epoch}")
print(f"Best Validation Loss  : {best_val_loss:.6f}")
print("=" * 60)

# Plotting
plt.style.use("seaborn-v0_8-paper")
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

ax.plot(
    epochs,
    train_loss,
    label="Training Loss (NT-Xent)",
    color="#1f77b4",
    linewidth=2,
    linestyle="--",
)
ax.plot(
    epochs,
    val_loss,
    label="Validation Loss (NT-Xent)",
    color="#ff7f0e",
    linewidth=2.5,
)

ax.axvline(
    x=best_epoch,
    color="#2ca02c",
    linestyle=":",
    linewidth=1.5,
    label=f"Optimal Checkpoint (Epoch {best_epoch})",
)
ax.scatter(
    [best_epoch],
    [best_val_loss],
    color="#2ca02c",
    s=80,
    zorder=5,
    edgecolor="black",
)

ax.set_title(
    "Member 2: NT-Xent Pre-Training Loss Convergence",
    fontsize=12,
    fontweight="bold",
)
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("NT-Xent Loss", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(frameon=True, facecolor="white", framealpha=0.9)

output_plot = os.path.join(PROJECT_ROOT, "loss_convergence.png")
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print(f"Saved plot to: {output_plot}\n")
plt.show()