import sys
from pathlib import Path

import torch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.gatv2_encoder import GATv2Encoder


print("======================================")
print("REAL DATA GATv2 TEST")
print("======================================")

# --------------------------------------------------
# Load Member 1 data
# --------------------------------------------------

print("\nLoading graph.pt...")

data = torch.load(
    PROJECT_ROOT / "graph.pt",
    weights_only=False
)

sequences = data["sequences"]

print("Total sequences:", len(sequences))

# --------------------------------------------------
# Get first sequence
# --------------------------------------------------

sequence = sequences[0]

snapshots = sequence["snapshots"]

print("Snapshots in first sequence:", len(snapshots))

# --------------------------------------------------
# Get first snapshot
# --------------------------------------------------

graph = snapshots[0]

print("\n======================================")
print("INPUT GRAPH")
print("======================================")

print("x shape:")
print(graph.x.shape)

print("edge_index shape:")
print(graph.edge_index.shape)

print("edge_attr shape:")
print(graph.edge_attr.shape)

# --------------------------------------------------
# Create GATv2
# --------------------------------------------------

gat_model = GATv2Encoder(
    in_channels=53,
    hidden_channels=128,
    heads=4,
    dropout=0.2
)

print("\nGATv2 model created.")

# --------------------------------------------------
# Forward pass
# --------------------------------------------------

print("\nRunning forward pass...")

with torch.no_grad():

    node_embeddings = gat_model(
        graph.x,
        graph.edge_index,
        graph.edge_attr
    )

# --------------------------------------------------
# Output
# --------------------------------------------------

print("\n======================================")
print("OUTPUT")
print("======================================")

print("Node embeddings shape:")
print(node_embeddings.shape)

print("\n======================================")
print("TEST RESULT")
print("======================================")

print("Real-data GATv2 forward pass successful!")

print("======================================")
