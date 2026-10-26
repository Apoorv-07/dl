import torch
import pickle

print("======================================")
print("CHECKING MEMBER 1 DATASET 2")
print("======================================")

# --------------------------------------------------
# 1. Load graph_validation.pt (Dataset 2)
# --------------------------------------------------

print("\nLoading graph_validation.pt...")

data = torch.load("graph_validation.pt", weights_only=False)

print("graph_validation.pt loaded successfully!")

print("\nTop-level type:")
print(type(data))

print("\nTop-level keys:")
print(data.keys())

# --------------------------------------------------
# 2. Sequences
# --------------------------------------------------

sequences = data["sequences"]

print("\n======================================")
print("SEQUENCE INFORMATION")
print("======================================")

print("Total sequences:", len(sequences))

# --------------------------------------------------
# 3. First sequence
# --------------------------------------------------

seq0 = sequences[0]

print("\nFirst sequence keys:")
print(seq0.keys())

print("\nSequence ID:")
print(seq0["sequence_id"])

print("\nLabel:")
print(seq0["label"])

print("\nWindow size:")
print(seq0["window_size_sec"], "seconds")

print("\nNumber of snapshots:")
print(seq0["num_snapshots"])

# --------------------------------------------------
# 4. Snapshot information
# --------------------------------------------------

snapshots = seq0["snapshots"]

print("\n======================================")
print("SNAPSHOT INFORMATION")
print("======================================")

for i, snap in enumerate(snapshots):

    print(f"\nSnapshot {i + 1}")

    print("x shape:")
    print(snap.x.shape)

    print("edge_index shape:")
    print(snap.edge_index.shape)

    print("edge_attr shape:")
    print(snap.edge_attr.shape)

# --------------------------------------------------
# 5. Feature columns
# --------------------------------------------------

print("\n======================================")
print("FEATURE INFORMATION")
print("======================================")

with open("feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

print("Number of features:")
print(len(feature_cols))

print("\nFeature columns:")

for i, feature in enumerate(feature_cols):
    print(f"{i}: {feature}")

# --------------------------------------------------
# 6. Important check
# --------------------------------------------------

print("\n======================================")
print("53-FEATURE CHECK")
print("======================================")

if len(feature_cols) == 53:
    print("SUCCESS: Feature count = 53")
else:
    print("WARNING: Feature count is NOT 53")
    print("Actual feature count:", len(feature_cols))

# --------------------------------------------------
# 7. Basic tensor consistency
# --------------------------------------------------

print("\n======================================")
print("TENSOR CONSISTENCY CHECK")
print("======================================")

first_snapshot = snapshots[0]

print("x feature dimension:",
      first_snapshot.x.shape[1])

print("edge feature dimension:",
      first_snapshot.edge_attr.shape[1])

if first_snapshot.x.shape[1] == len(feature_cols):
    print("SUCCESS: x dimension matches feature_cols")
else:
    print("WARNING: x dimension does NOT match feature_cols")

if first_snapshot.edge_attr.shape[1] == len(feature_cols):
    print("SUCCESS: edge_attr dimension matches feature_cols")
else:
    print("WARNING: edge_attr dimension does NOT match feature_cols")

print("\n======================================")
print("DATASET 2 CHECK COMPLETE")
print("======================================")