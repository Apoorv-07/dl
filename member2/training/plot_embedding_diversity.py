import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.gatv2_encoder import GATv2Encoder
from models.global_pooling import global_mean_pooling
from models.temporal_transformer import TemporalTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data and Model
data_path = os.path.join(PROJECT_ROOT, "graph.pt")
ckpt_path = os.path.join(
    PROJECT_ROOT, "checkpoints_validation", "best_model.pt"
)

data = torch.load(data_path, weights_only=False)
sequences = data["sequences"]

gat = GATv2Encoder(
    in_channels=53, hidden_channels=128, heads=4, dropout=0.2
).to(DEVICE)
transformer = TemporalTransformer(
    input_dim=128, num_heads=4, num_layers=2, dropout=0.1
).to(DEVICE)

ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
gat.load_state_dict(ckpt["gat_state_dict"])
transformer.load_state_dict(ckpt["transformer_state_dict"])

gat.eval()
transformer.eval()

# Generate h embeddings
embeddings = []
with torch.no_grad():
  for seq in sequences[:500]:  # Evaluate on sample subset
    snapshot_embeds = []
    for graph in seq["snapshots"]:
      x = graph.x.to(DEVICE)
      edge_index = graph.edge_index.to(DEVICE)
      edge_attr = graph.edge_attr.to(DEVICE)

      if x is not None:
        x = torch.clamp(x, min=-5.0, max=5.0)
      if edge_attr is not None:
        edge_attr = torch.clamp(edge_attr, min=-5.0, max=5.0)

      node_e = gat(x, edge_index, edge_attr)
      graph_e = global_mean_pooling(node_e)
      snapshot_embeds.append(graph_e)

    temporal_in = torch.cat(snapshot_embeds, dim=0).unsqueeze(0)
    h = transformer(temporal_in)
    h = F.normalize(h, p=2, dim=-1)
    embeddings.append(h)

H = torch.cat(embeddings, dim=0)  # [500, 128]

# Compute Pairwise Cosine Similarity
sim_matrix = torch.mm(H, H.T).cpu().numpy()
triu_indices = np.triu_indices_from(sim_matrix, k=1)
pairwise_sims = sim_matrix[triu_indices]

print("=" * 60)
print("EMBEDDING UNIFORMITY & DIVERSITY STATS")
print("=" * 60)
print(f"Mean Pairwise Cosine Similarity   : {np.mean(pairwise_sims):.6f}")
print(f"Median Pairwise Cosine Similarity : {np.median(pairwise_sims):.6f}")
print(f"Min Pairwise Cosine Similarity    : {np.min(pairwise_sims):.6f}")
print(f"Max Pairwise Cosine Similarity    : {np.max(pairwise_sims):.6f}")
print("=" * 60)

# Plotting Histogram
plt.style.use("seaborn-v0_8-paper")
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

ax.hist(
    pairwise_sims, bins=50, color="#1f77b4", edgecolor="black", alpha=0.7
)
ax.axvline(
    x=np.median(pairwise_sims),
    color="#d62728",
    linestyle="--",
    linewidth=2,
    label=f"Median Similarity ({np.median(pairwise_sims):.3f})",
)

ax.set_title(
    "Backbone Representation ($h$) Pairwise Similarity Distribution",
    fontsize=12,
    fontweight="bold",
)
ax.set_xlabel(
    "Pairwise Cosine Similarity $\\cos(h_i, h_j)$", fontsize=10
)
ax.set_ylabel("Frequency", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(frameon=True, facecolor="white", framealpha=0.9)

output_plot = os.path.join(PROJECT_ROOT, "embedding_diversity.png")
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print(f"Saved plot to: {output_plot}\n")
plt.show()