import os
import sys
import torch
import numpy as np

# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.member2_model import Member2FoundationModel


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "graph_validation.pt"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints_validation",
    "best_model.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
print("=" * 70)
print("EMBEDDING COLLAPSE CHECK")
print("=" * 70)

data = torch.load(
    DATASET_PATH,
    map_location="cpu",
    weights_only=False
)

sequences = data["sequences"]

print(f"\nDevice: {DEVICE}")
print(f"Total sequences: {len(sequences)}")


# ---------------------------------------------------------
# LOAD CHECKPOINT
# ---------------------------------------------------------
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Best epoch: {checkpoint['best_epoch']}")
print(
    f"Best validation loss: "
    f"{checkpoint['best_validation_loss']}"
)


# ---------------------------------------------------------
# CREATE MODEL
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# LOAD GAT
# ---------------------------------------------------------
gat_state_dict = checkpoint["gat_state_dict"]

gat1_state_dict = {
    key.replace("gat1.", "", 1): value
    for key, value in gat_state_dict.items()
    if key.startswith("gat1.")
}

gat2_state_dict = {
    key.replace("gat2.", "", 1): value
    for key, value in gat_state_dict.items()
    if key.startswith("gat2.")
}

model.gat1.load_state_dict(gat1_state_dict)
model.gat2.load_state_dict(gat2_state_dict)

model.temporal_transformer.load_state_dict(
    checkpoint["transformer_state_dict"]
)

model.projection_head.load_state_dict(
    checkpoint["projection_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("\nModel loaded successfully.")


# ---------------------------------------------------------
# GENERATE H EMBEDDINGS
# ---------------------------------------------------------
print("\nGenerating h embeddings...")

embeddings = []
labels = []

with torch.no_grad():

    for i, sequence in enumerate(sequences):

        snapshots = [
            snapshot.to(DEVICE)
            for snapshot in sequence["snapshots"]
        ]

        output = model.encode_sequence(snapshots)

        h = output["h"]

        embeddings.append(
            h.squeeze(0).cpu()
        )

        labels.append(
            int(sequence["label"])
        )

        if (i + 1) % 500 == 0:
            print(
                f"Processed {i + 1}/{len(sequences)}"
            )


embeddings = torch.stack(embeddings)

labels = np.array(labels)

print(
    f"\nEmbedding shape: {embeddings.shape}"
)


# ---------------------------------------------------------
# SPLIT
# ---------------------------------------------------------
benign = embeddings[labels == 0]
attack = embeddings[labels == 1]


# ---------------------------------------------------------
# BASIC STATISTICS
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("EMBEDDING STATISTICS")
print("=" * 70)

print("\nOverall:")
print(
    f"Mean absolute value : "
    f"{embeddings.abs().mean().item():.10f}"
)

print(
    f"Standard deviation  : "
    f"{embeddings.std().item():.10f}"
)

print(
    f"Minimum             : "
    f"{embeddings.min().item():.10f}"
)

print(
    f"Maximum             : "
    f"{embeddings.max().item():.10f}"
)

print(
    f"Mean L2 norm        : "
    f"{torch.norm(embeddings, dim=1).mean().item():.10f}"
)

print(
    f"Std L2 norm         : "
    f"{torch.norm(embeddings, dim=1).std().item():.10f}"
)


# ---------------------------------------------------------
# BENIGN STATISTICS
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("BENIGN EMBEDDINGS")
print("=" * 70)

print(
    f"Mean absolute value : "
    f"{benign.abs().mean().item():.10f}"
)

print(
    f"Standard deviation  : "
    f"{benign.std().item():.10f}"
)

print(
    f"Mean L2 norm        : "
    f"{torch.norm(benign, dim=1).mean().item():.10f}"
)

print(
    f"Std L2 norm         : "
    f"{torch.norm(benign, dim=1).std().item():.10f}"
)


# ---------------------------------------------------------
# ATTACK STATISTICS
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("ATTACK EMBEDDINGS")
print("=" * 70)

print(
    f"Mean absolute value : "
    f"{attack.abs().mean().item():.10f}"
)

print(
    f"Standard deviation  : "
    f"{attack.std().item():.10f}"
)

print(
    f"Mean L2 norm        : "
    f"{torch.norm(attack, dim=1).mean().item():.10f}"
)

print(
    f"Std L2 norm         : "
    f"{torch.norm(attack, dim=1).std().item():.10f}"
)


# ---------------------------------------------------------
# PAIRWISE COSINE SIMILARITY
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("PAIRWISE COSINE SIMILARITY")
print("=" * 70)

# Normalize embeddings
normalized = torch.nn.functional.normalize(
    embeddings,
    p=2,
    dim=1
)

# Random sample to avoid huge NxN matrix
torch.manual_seed(42)

sample_size = min(500, len(normalized))

indices = torch.randperm(
    len(normalized)
)[:sample_size]

sample = normalized[indices]

similarity_matrix = sample @ sample.T


# Remove diagonal
mask = ~torch.eye(
    sample_size,
    dtype=torch.bool
)

pairwise_similarities = similarity_matrix[mask]

print(
    f"Sample size: {sample_size}"
)

print(
    f"Mean pairwise cosine similarity: "
    f"{pairwise_similarities.mean().item():.10f}"
)

print(
    f"Median pairwise cosine similarity: "
    f"{pairwise_similarities.median().item():.10f}"
)

print(
    f"Minimum pairwise cosine similarity: "
    f"{pairwise_similarities.min().item():.10f}"
)

print(
    f"Maximum pairwise cosine similarity: "
    f"{pairwise_similarities.max().item():.10f}"
)


# ---------------------------------------------------------
# BENIGN VS ATTACK MEAN EMBEDDING
# ---------------------------------------------------------
benign_mean = benign.mean(dim=0)
attack_mean = attack.mean(dim=0)

mean_cosine = torch.nn.functional.cosine_similarity(
    benign_mean.unsqueeze(0),
    attack_mean.unsqueeze(0)
).item()

mean_euclidean = torch.norm(
    benign_mean - attack_mean
).item()

print("\n" + "=" * 70)
print("BENIGN vs ATTACK CENTERS")
print("=" * 70)

print(
    f"Cosine similarity between means: "
    f"{mean_cosine:.10f}"
)

print(
    f"Euclidean distance between means: "
    f"{mean_euclidean:.10f}"
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
