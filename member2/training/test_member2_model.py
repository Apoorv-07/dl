import sys
import os

# Allow imports from project root
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch

from models.member2_model import Member2FoundationModel


print("======================================")
print("MEMBER 2 INTEGRATION MODEL TEST")
print("======================================")

print()
print("Loading graph.pt...")

data = torch.load(
    "graph.pt",
    weights_only=False
)

sequences = data["sequences"]

print("Total sequences:", len(sequences))

# --------------------------------------------------
# SELECT FIRST REAL SEQUENCE
# --------------------------------------------------

sequence_data = sequences[0]

snapshots = sequence_data["snapshots"]

print("Sequence ID:", sequence_data["sequence_id"])
print("Label:", sequence_data["label"])
print("Number of snapshots:", len(snapshots))

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = Member2FoundationModel(
    in_channels=53,
    hidden_channels=128,
    heads=4,
    projection_dim=64,
    dropout=0.2,
)

model = model.to(device)

model.eval()

# Move snapshots to device
snapshots = [
    snapshot.to(device)
    for snapshot in snapshots
]

# --------------------------------------------------
# NORMAL ENCODING
# --------------------------------------------------

print()
print("======================================")
print("TESTING encode_sequence()")
print("======================================")

with torch.no_grad():

    output = model.encode_sequence(
        snapshots,
        return_attention=False,
    )

print()
print("Returned keys:")
print(output.keys())

print()
print("h shape:")
print(output["h"].shape)

print()
print("z shape:")
print(output["z"].shape)

# --------------------------------------------------
# ATTENTION ENCODING
# --------------------------------------------------

print()
print("======================================")
print("TESTING ATTENTION OUTPUT")
print("======================================")

with torch.no_grad():

    attention_output = model.encode_sequence(
        snapshots,
        return_attention=True,
    )

print()
print("Returned keys:")
print(attention_output.keys())

print()
print("h shape:")
print(attention_output["h"].shape)

print()
print("z shape:")
print(attention_output["z"].shape)

print()
print("Number of node attention tensors:")
print(
    len(
        attention_output["node_attention"]
    )
)

print()
print("Number of edge attention tensors:")
print(
    len(
        attention_output["edge_attention"]
    )
)

# --------------------------------------------------
# VERIFY ATTENTION LENGTHS
# --------------------------------------------------

print()
print("======================================")
print("VERIFYING ATTENTION DIMENSIONS")
print("======================================")

for i, snapshot in enumerate(snapshots):

    node_att = attention_output[
        "node_attention"
    ][i]

    edge_att = attention_output[
        "edge_attention"
    ][i]

    print()
    print(f"Snapshot {i + 1}")

    print(
        "Nodes:",
        snapshot.x.size(0),
        "| Node attention:",
        node_att.shape
    )

    print(
        "Edges:",
        snapshot.edge_index.size(1),
        "| Edge attention:",
        edge_att.shape
    )

print()
print("======================================")
print("TEST COMPLETE")
print("======================================")