import os
import sys
import torch

# --------------------------------------------------
# 1. Project root
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

# --------------------------------------------------
# 2. Import model
# --------------------------------------------------

from models.member2_model import Member2FoundationModel


print("======================================")
print("MEMBER 2 - DATASET 2 EVALUATION")
print("======================================")

# --------------------------------------------------
# 3. Device
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", DEVICE)

# --------------------------------------------------
# 4. Paths
# --------------------------------------------------

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "graph_validation.pt"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints_validation",
    "best_model.pt"
)

# --------------------------------------------------
# 5. Load Dataset 2
# --------------------------------------------------

print("\nLoading Dataset 2...")

data = torch.load(
    DATASET_PATH,
    weights_only=False
)

sequences = data["sequences"]

print("Dataset 2 loaded successfully!")
print("Total sequences:", len(sequences))

# --------------------------------------------------
# 6. Check labels
# --------------------------------------------------

labels = [
    int(sequence["label"])
    for sequence in sequences
]

num_benign = sum(
    label == 0
    for label in labels
)

num_attack = sum(
    label == 1
    for label in labels
)

print("\n======================================")
print("DATASET 2 LABEL INFORMATION")
print("======================================")

print("Benign sequences :", num_benign)
print("Attack sequences :", num_attack)

# --------------------------------------------------
# 7. Create model
# --------------------------------------------------

print("\n======================================")
print("CREATING MODEL")
print("======================================")

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

model = model.to(DEVICE)

# --------------------------------------------------
# 8. Load checkpoint
# --------------------------------------------------

print("\nLoading best checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

# --------------------------------------------------
# Load GAT layers
# --------------------------------------------------

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

model.gat1.load_state_dict(
    gat1_state_dict
)

model.gat2.load_state_dict(
    gat2_state_dict
)

# --------------------------------------------------
# Load Transformer
# --------------------------------------------------

model.temporal_transformer.load_state_dict(
    checkpoint["transformer_state_dict"]
)

# --------------------------------------------------
# Load Projection Head
# --------------------------------------------------

model.projection_head.load_state_dict(
    checkpoint["projection_state_dict"]
)

print("Best checkpoint loaded successfully!")

print(
    "Checkpoint epoch:",
    checkpoint.get("epoch")
)

print(
    "Best epoch:",
    checkpoint.get("best_epoch")
)

print(
    "Best validation loss:",
    checkpoint.get("best_validation_loss")
)

# --------------------------------------------------
# 9. Freeze model
# --------------------------------------------------

model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False

print("\nModel is frozen.")
print("Inference mode enabled.")

# --------------------------------------------------
# 10. Generate embeddings
# --------------------------------------------------

print("\n======================================")
print("GENERATING EMBEDDINGS")
print("======================================")

embeddings_h = []
embeddings_z = []

sequence_labels = []
sequence_ids = []

with torch.no_grad():

    for i, sequence in enumerate(sequences):

        # --------------------------------------------------
        # Get snapshots from the sequence
        # --------------------------------------------------

        snapshots = sequence["snapshots"]

        # --------------------------------------------------
        # Move every snapshot to the same device
        # as the model
        # --------------------------------------------------

        gpu_snapshots = []

        for snapshot in snapshots:

            snapshot = snapshot.to(DEVICE)

            gpu_snapshots.append(snapshot)

        # --------------------------------------------------
        # Generate embeddings
        # --------------------------------------------------

        output = model.encode_sequence(
            gpu_snapshots
        )

        h = output["h"]
        z = output["z"]

        # --------------------------------------------------
        # Move embeddings back to CPU
        # --------------------------------------------------

        h = h.detach().cpu()
        z = z.detach().cpu()

        embeddings_h.append(h)
        embeddings_z.append(z)

        # --------------------------------------------------
        # Store labels and sequence IDs
        # --------------------------------------------------

        sequence_labels.append(
            int(sequence["label"])
        )

        sequence_ids.append(
            sequence["sequence_id"]
        )

        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if (i + 1) % 100 == 0:

            print(
                f"Processed {i + 1}/{len(sequences)} sequences"
            )

# --------------------------------------------------
# 11. Stack embeddings
# --------------------------------------------------

embeddings_h = torch.cat(
    embeddings_h,
    dim=0
)

embeddings_z = torch.cat(
    embeddings_z,
    dim=0
)

labels_tensor = torch.tensor(
    sequence_labels,
    dtype=torch.long
)

# --------------------------------------------------
# 12. Final results
# --------------------------------------------------

print("\n======================================")
print("EMBEDDING RESULTS")
print("======================================")

print("h embedding shape:")
print(embeddings_h.shape)

print("\nz embedding shape:")
print(embeddings_z.shape)

print("\nLabels shape:")
print(labels_tensor.shape)

# --------------------------------------------------
# 13. Dimension check
# --------------------------------------------------

print("\n======================================")
print("DIMENSION CHECK")
print("======================================")

if embeddings_h.shape[1] == 128:

    print("SUCCESS: h dimension = 128")

else:

    print(
        "WARNING: unexpected h dimension:",
        embeddings_h.shape[1]
    )

if embeddings_z.shape[1] == 64:

    print("SUCCESS: z dimension = 64")

else:

    print(
        "WARNING: unexpected z dimension:",
        embeddings_z.shape[1]
    )

# --------------------------------------------------
# 14. Final label counts
# --------------------------------------------------

print("\n======================================")
print("FINAL LABEL COUNTS")
print("======================================")

print(
    "Benign (0):",
    int((labels_tensor == 0).sum())
)

print(
    "Attack (1):",
    int((labels_tensor == 1).sum())
)

# --------------------------------------------------
# 15. Sample embedding
# --------------------------------------------------

print("\n======================================")
print("SAMPLE EMBEDDING")
print("======================================")

print(
    "Sequence ID:",
    sequence_ids[0]
)

print(
    "Label:",
    sequence_labels[0]
)

print("\nFirst h embedding:")
print(embeddings_h[0])

print("\nFirst z embedding:")
print(embeddings_z[0])

print("\n======================================")
print("DATASET 2 EVALUATION COMPLETE")
print("======================================")