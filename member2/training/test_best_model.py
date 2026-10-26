import torch
import sys
import os


# ==================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)


from models.member2_model import Member2FoundationModel


print("======================================")
print("TESTING BEST MEMBER 2 MODEL")
print("======================================")


# ==================================================
# 1. DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)


# ==================================================
# 2. LOAD graph.pt
# ==================================================

print("\nLoading graph.pt...")

data = torch.load(
    "graph.pt",
    weights_only=False
)

sequences = data["sequences"]

print("Total sequences:", len(sequences))


# ==================================================
# 3. TAKE FIRST REAL SEQUENCE
# ==================================================

sequence = sequences[0]

print("\nSequence ID:", sequence["sequence_id"])
print("Label:", sequence["label"])
print("Number of snapshots:", sequence["num_snapshots"])


# ==================================================
# 4. CREATE MEMBER 2 MODEL
# ==================================================

print("\nCreating Member 2 model...")

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

model = model.to(device)


# ==================================================
# 5. LOAD BEST CHECKPOINT
# ==================================================

print("\nLoading best checkpoint...")

checkpoint = torch.load(
    "checkpoints_validation/best_model.pt",
    map_location=device,
    weights_only=False
)


# ==================================================
# 6. LOAD GAT WEIGHTS
# ==================================================

gat_state_dict = checkpoint["gat_state_dict"]


# -------------------------------
# GAT Layer 1
# -------------------------------

gat1_state_dict = {
    key.replace("gat1.", "", 1): value
    for key, value in gat_state_dict.items()
    if key.startswith("gat1.")
}


# -------------------------------
# GAT Layer 2
# -------------------------------

gat2_state_dict = {
    key.replace("gat2.", "", 1): value
    for key, value in gat_state_dict.items()
    if key.startswith("gat2.")
}


# -------------------------------
# Load GAT weights
# -------------------------------

model.gat1.load_state_dict(
    gat1_state_dict
)

model.gat2.load_state_dict(
    gat2_state_dict
)


# ==================================================
# 7. LOAD TRANSFORMER WEIGHTS
# ==================================================

model.temporal_transformer.load_state_dict(
    checkpoint["transformer_state_dict"]
)


# ==================================================
# 8. LOAD PROJECTION HEAD WEIGHTS
# ==================================================

model.projection_head.load_state_dict(
    checkpoint["projection_state_dict"]
)


print("Best checkpoint loaded successfully!")

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    "Best epoch:",
    checkpoint["best_epoch"]
)

print(
    "Best validation loss:",
    checkpoint["best_validation_loss"]
)


# ==================================================
# 9. EVALUATION MODE
# ==================================================

model.eval()


# ==================================================
# 10. MOVE SEQUENCE TO GPU
# ==================================================

for snapshot in sequence["snapshots"]:

    snapshot.x = snapshot.x.to(device)

    snapshot.edge_index = (
        snapshot.edge_index.to(device)
    )

    snapshot.edge_attr = (
        snapshot.edge_attr.to(device)
    )


# ==================================================
# 11. RUN INFERENCE
# ==================================================

print("\nRunning inference on first sequence...")

with torch.no_grad():

    output = model.encode_sequence(
        sequence["snapshots"]
    )


# ==================================================
# 12. DISPLAY OUTPUT
# ==================================================

print("\n======================================")
print("MODEL OUTPUT")
print("======================================")

print(
    "Output keys:",
    output.keys()
)


print("\nh shape:")

print(
    output["h"].shape
)


print("\nz shape:")

print(
    output["z"].shape
)


# ==================================================
# 13. SHAPE CHECK
# ==================================================

print("\n======================================")
print("SHAPE CHECK")
print("======================================")


if output["h"].shape == (1, 128):

    print(
        "SUCCESS: h shape = [1, 128]"
    )

else:

    print(
        "WARNING: Unexpected h shape:",
        output["h"].shape
    )


if output["z"].shape == (1, 64):

    print(
        "SUCCESS: z shape = [1, 64]"
    )

else:

    print(
        "WARNING: Unexpected z shape:",
        output["z"].shape
    )


# ==================================================
# 14. FINISH
# ==================================================

print("\n======================================")
print("TEST COMPLETE")
print("======================================")