import sys
from pathlib import Path

import torch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.dummy_data import create_dummy_sequence
from models.gatv2_encoder import GATv2Encoder
from models.global_pooling import global_mean_pooling


# --------------------------------------------------
# 1. Create dummy temporal graph sequence
# --------------------------------------------------

sequence = create_dummy_sequence()


# --------------------------------------------------
# 2. Create GATv2 encoder
# --------------------------------------------------

model = GATv2Encoder(
    in_channels=53,
    hidden_channels=128,
    heads=4
)


# --------------------------------------------------
# 3. Process all 5 graph snapshots
# --------------------------------------------------

print("Testing all 5 snapshots...\n")

graph_vectors = []


for i, graph in enumerate(sequence):

    # GATv2:
    # [10, 53] -> [10, 128]

    node_embeddings = model(
        graph.x,
        graph.edge_index
    )

    # Global Mean Pooling:
    # [10, 128] -> [1, 128]

    graph_embedding = global_mean_pooling(
        node_embeddings
    )

    graph_vectors.append(graph_embedding)

    print(f"Snapshot {i + 1}")
    print("Node embeddings:", node_embeddings.shape)
    print("Graph embedding:", graph_embedding.shape)
    print()


# --------------------------------------------------
# 4. Combine all 5 graph embeddings
# --------------------------------------------------

temporal_sequence = torch.cat(
    graph_vectors,
    dim=0
)


# --------------------------------------------------
# 5. Final output
# --------------------------------------------------

print("====================================")
print("Final temporal sequence:")
print(temporal_sequence.shape)
print("====================================")