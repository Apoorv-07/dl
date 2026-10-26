import sys
from pathlib import Path

import torch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.dummy_data import create_dummy_sequence
from models.gatv2_encoder import GATv2Encoder
from models.global_pooling import global_mean_pooling
from models.temporal_transformer import TemporalTransformer


# --------------------------------------------------
# 1. Create dummy 5-snapshot sequence
# --------------------------------------------------

sequence = create_dummy_sequence()


# --------------------------------------------------
# 2. Create GATv2 encoder
# --------------------------------------------------

gat_model = GATv2Encoder(
    in_channels=53,
    hidden_channels=128,
    heads=4
)


# --------------------------------------------------
# 3. Create graph-level vectors
# --------------------------------------------------

graph_vectors = []

for graph in sequence:

    # GATv2
    node_embeddings = gat_model(
        graph.x,
        graph.edge_index
    )

    # Global Mean Pooling
    graph_embedding = global_mean_pooling(
        node_embeddings
    )

    graph_vectors.append(graph_embedding)


# --------------------------------------------------
# 4. Combine the 5 snapshots
# --------------------------------------------------

temporal_sequence = torch.cat(
    graph_vectors,
    dim=0
)

print("Before batch dimension:")
print(temporal_sequence.shape)


# --------------------------------------------------
# 5. Add batch dimension
# --------------------------------------------------

temporal_sequence = temporal_sequence.unsqueeze(0)

print("After batch dimension:")
print(temporal_sequence.shape)


# --------------------------------------------------
# 6. Create Temporal Transformer
# --------------------------------------------------

transformer = TemporalTransformer(
    input_dim=128,
    num_heads=4,
    num_layers=2
)


# --------------------------------------------------
# 7. Pass through Transformer
# --------------------------------------------------

h = transformer(
    temporal_sequence
)


# --------------------------------------------------
# 8. Print final representation
# --------------------------------------------------

print("\nBackbone representation h:")
print(h.shape)