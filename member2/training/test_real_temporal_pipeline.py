import sys
from pathlib import Path

import torch

# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --------------------------------------------------
# Import our models
# --------------------------------------------------

from models.gatv2_encoder import GATv2Encoder
from models.global_pooling import global_mean_pooling
from models.temporal_transformer import TemporalTransformer


print("======================================")
print("REAL TEMPORAL PIPELINE TEST")
print("======================================")

# --------------------------------------------------
# 1. Load Member 1 real data
# --------------------------------------------------

print("\nLoading graph.pt...")

data = torch.load(
    PROJECT_ROOT / "graph.pt",
    weights_only=False
)

sequences = data["sequences"]

print("Total sequences:", len(sequences))

# --------------------------------------------------
# 2. Select first real sequence
# --------------------------------------------------

sequence = sequences[0]

snapshots = sequence["snapshots"]

print("Sequence ID:", sequence["sequence_id"])
print("Label:", sequence["label"])
print("Number of snapshots:", len(snapshots))

# --------------------------------------------------
# 3. Create GATv2
# --------------------------------------------------

gat_model = GATv2Encoder(
    in_channels=53,
    hidden_channels=128,
    heads=4,
    dropout=0.2
)

# --------------------------------------------------
# 4. Create Temporal Transformer
# --------------------------------------------------

transformer = TemporalTransformer(
    input_dim=128,
    num_heads=4,
    num_layers=2,
    dropout=0.1
)

# --------------------------------------------------
# 5. Process all five snapshots
# --------------------------------------------------

graph_embeddings = []

print("\n======================================")
print("PROCESSING SNAPSHOTS")
print("======================================")

with torch.no_grad():

    for i, graph in enumerate(snapshots):

        print(f"\nSnapshot {i + 1}")

        print("Input x:", graph.x.shape)
        print("Input edge_index:", graph.edge_index.shape)
        print("Input edge_attr:", graph.edge_attr.shape)

        # ------------------------------------------
        # GATv2
        # ------------------------------------------

        node_embeddings = gat_model(
            graph.x,
            graph.edge_index,
            graph.edge_attr
        )

        print(
            "GATv2 output:",
            node_embeddings.shape
        )

        # ------------------------------------------
        # Global Mean Pooling
        # ------------------------------------------

        graph_embedding = global_mean_pooling(
            node_embeddings
        )

        print(
            "Graph embedding:",
            graph_embedding.shape
        )

        graph_embeddings.append(graph_embedding)

# --------------------------------------------------
# 6. Create temporal sequence
# --------------------------------------------------

print("\n======================================")
print("TEMPORAL SEQUENCE")
print("======================================")

temporal_sequence = torch.cat(
    graph_embeddings,
    dim=0
)

print(
    "Before batch dimension:",
    temporal_sequence.shape
)

# Expected:
# [5, 128]

temporal_sequence = temporal_sequence.unsqueeze(0)

print(
    "After batch dimension:",
    temporal_sequence.shape
)

# Expected:
# [1, 5, 128]

# --------------------------------------------------
# 7. Temporal Transformer
# --------------------------------------------------

print("\n======================================")
print("TEMPORAL TRANSFORMER")
print("======================================")

with torch.no_grad():

    h = transformer(
        temporal_sequence
    )

print(
    "Backbone representation h:",
    h.shape
)

# Expected:
# [1, 128]

# --------------------------------------------------
# 8. Final result
# --------------------------------------------------

print("\n======================================")
print("TEST RESULT")
print("======================================")

if h.shape == torch.Size([1, 128]):

    print(
        "SUCCESS: Real 5-snapshot sequence "
        "passed through GATv2 + Pooling + Transformer."
    )

else:

    print(
        "WARNING: Unexpected h shape:",
        h.shape
    )

print("\n======================================")
print("REAL TEMPORAL PIPELINE TEST COMPLETE")
print("======================================")