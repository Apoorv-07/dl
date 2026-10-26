import os
import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.member2_model import Member2FoundationModel

DATASET_PATH = os.path.join(PROJECT_ROOT, "graph_validation.pt")
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "checkpoints_validation", "best_model.pt"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 5
SEED = 42

print("=" * 70)
print("Z EMBEDDING KNN EVALUATION")
print("=" * 70)

data = torch.load(
    DATASET_PATH,
    map_location="cpu",
    weights_only=False
)

sequences = data["sequences"]

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

model = Member2FoundationModel(
    in_channels=53,
    hidden_channels=128,
    heads=4,
    transformer_heads=4,
    transformer_layers=2,
    h_dim=128,
    z_dim=64,
    dropout=0.2
)

gat_state = checkpoint["gat_state_dict"]

model.gat1.load_state_dict({
    k.replace("gat1.", "", 1): v
    for k, v in gat_state.items()
    if k.startswith("gat1.")
})

model.gat2.load_state_dict({
    k.replace("gat2.", "", 1): v
    for k, v in gat_state.items()
    if k.startswith("gat2.")
})

model.temporal_transformer.load_state_dict(
    checkpoint["transformer_state_dict"]
)

model.projection_head.load_state_dict(
    checkpoint["projection_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print(f"Device: {DEVICE}")
print(f"Checkpoint epoch: {checkpoint['epoch']}")

# ---------------------------------------------------------
# GENERATE Z
# ---------------------------------------------------------
embeddings = []
labels = []

print("\nGenerating z embeddings...")

with torch.no_grad():

    for i, sequence in enumerate(sequences):

        snapshots = [
            snapshot.to(DEVICE)
            for snapshot in sequence["snapshots"]
        ]

        output = model.encode_sequence(snapshots)

        embeddings.append(
            output["z"].squeeze(0).cpu()
        )

        labels.append(int(sequence["label"]))

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(sequences)}")

embeddings = torch.stack(embeddings).numpy()
labels = np.array(labels)

print(f"\nZ shape: {embeddings.shape}")

# ---------------------------------------------------------
# LEAKAGE-FREE SPLIT
# ---------------------------------------------------------
rng = np.random.default_rng(SEED)

benign = np.where(labels == 0)[0]
attack = np.where(labels == 1)[0]

rng.shuffle(benign)

split = int(len(benign) * 0.80)

memory_idx = benign[:split]
benign_test_idx = benign[split:]

test_idx = np.concatenate([
    benign_test_idx,
    attack
])

# ---------------------------------------------------------
# L2 NORMALIZATION
# ---------------------------------------------------------
def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)

memory = normalize(embeddings[memory_idx])
test = normalize(embeddings[test_idx])

test_labels = labels[test_idx]

# ---------------------------------------------------------
# COSINE KNN
# ---------------------------------------------------------
similarity = test @ memory.T
distance = 1.0 - similarity

nearest = np.partition(
    distance,
    K - 1,
    axis=1
)[:, :K]

scores = nearest.mean(axis=1)

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------
benign_scores = scores[test_labels == 0]
attack_scores = scores[test_labels == 1]

roc_auc = roc_auc_score(
    test_labels,
    scores
)

pr_auc = average_precision_score(
    test_labels,
    scores
)

print("\n" + "=" * 70)
print("Z KNN RESULTS")
print("=" * 70)

print("\nBenign:")
print(f"Mean   : {benign_scores.mean():.10f}")
print(f"Median : {np.median(benign_scores):.10f}")
print(f"95th   : {np.percentile(benign_scores, 95):.10f}")
print(f"99th   : {np.percentile(benign_scores, 99):.10f}")

print("\nAttack:")
print(f"Mean   : {attack_scores.mean():.10f}")
print(f"Median : {np.median(attack_scores):.10f}")
print(f"95th   : {np.percentile(attack_scores, 95):.10f}")
print(f"99th   : {np.percentile(attack_scores, 99):.10f}")

print("\n" + "=" * 70)
print("THRESHOLD-FREE PERFORMANCE")
print("=" * 70)

print(f"ROC-AUC : {roc_auc:.6f}")
print(f"PR-AUC  : {pr_auc:.6f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)