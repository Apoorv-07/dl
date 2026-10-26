import os
import sys
import torch

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from models.member2_model import Member2FoundationModel


# --------------------------------------------------
# PATHS
# --------------------------------------------------

TRAINING_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "checkpoints_validation",
    "epoch_5.pt"
)

FINAL_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "checkpoints_validation",
    "member2_dataset1_final_model.pt"
)


# --------------------------------------------------
# START
# --------------------------------------------------

print("======================================")
print("BUILDING MEMBER 2 DATASET 1 FINAL MODEL")
print("======================================")

print()
print("Training checkpoint:")
print(TRAINING_CHECKPOINT)

print()
print("Final checkpoint:")
print(FINAL_CHECKPOINT)


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("Device:")
print(device)


# --------------------------------------------------
# LOAD TRAINING CHECKPOINT
# --------------------------------------------------

print()
print("Loading epoch-5 checkpoint...")

checkpoint = torch.load(
    TRAINING_CHECKPOINT,
    map_location=device,
    weights_only=False
)

print("Epoch-5 checkpoint loaded successfully.")


# --------------------------------------------------
# DISPLAY CHECKPOINT INFORMATION
# --------------------------------------------------

print()
print("Checkpoint information:")

print("Epoch:", checkpoint["epoch"])
print("Training loss:", checkpoint["train_loss"])
print("Validation loss:", checkpoint["validation_loss"])
print("Best validation loss:", checkpoint["best_validation_loss"])
print("Best epoch:", checkpoint["best_epoch"])


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

print()
print("Creating Member2FoundationModel...")

model = Member2FoundationModel(
    in_channels=53,
    hidden_channels=128,
    heads=4,
    transformer_heads=4,
    transformer_layers=2,
    h_dim=128,
    z_dim=64,
    dropout=0.2,
)

model = model.to(device)


# --------------------------------------------------
# LOAD GAT WEIGHTS
# --------------------------------------------------

print()
print("Loading GAT weights...")

gat_state_dict = checkpoint["gat_state_dict"]

# Separate the two GAT layers.
gat1_state_dict = {
    key[len("gat1."):]: value
    for key, value in gat_state_dict.items()
    if key.startswith("gat1.")
}

gat2_state_dict = {
    key[len("gat2."):]: value
    for key, value in gat_state_dict.items()
    if key.startswith("gat2.")
}

model.gat1.load_state_dict(
    gat1_state_dict
)

model.gat2.load_state_dict(
    gat2_state_dict
)

print("GAT1 weights loaded.")
print("GAT2 weights loaded.")


# --------------------------------------------------
# LOAD TRANSFORMER WEIGHTS
# --------------------------------------------------

print()
print("Loading Transformer weights...")

model.temporal_transformer.load_state_dict(
    checkpoint["transformer_state_dict"]
)

print("Transformer weights loaded.")


# --------------------------------------------------
# LOAD PROJECTION HEAD WEIGHTS
# --------------------------------------------------

print()
print("Loading projection-head weights...")

model.projection_head.load_state_dict(
    checkpoint["projection_state_dict"]
)

print("Projection-head weights loaded.")


# --------------------------------------------------
# EVALUATION MODE
# --------------------------------------------------

model.eval()


# --------------------------------------------------
# SAVE COMPLETE MODEL
# --------------------------------------------------

print()
print("Saving complete model object...")

torch.save(
    model,
    FINAL_CHECKPOINT
)

print("Model saved successfully.")


# --------------------------------------------------
# RELOAD TO VERIFY
# --------------------------------------------------

print()
print("Reloading saved model...")

reloaded_model = torch.load(
    FINAL_CHECKPOINT,
    map_location=device,
    weights_only=False
)

reloaded_model = reloaded_model.to(device)
reloaded_model.eval()

print("Reload successful.")


# --------------------------------------------------
# FINAL INFORMATION
# --------------------------------------------------

print()
print("======================================")
print("FINAL MODEL INFORMATION")
print("======================================")

print()
print("Model type:")
print(type(reloaded_model))

print()
print("Model metadata:")

try:
    print(reloaded_model.metadata())
except Exception:
    print("metadata() unavailable")


print()
print("======================================")
print("FINAL CHECKPOINT BUILD COMPLETE")
print("======================================")