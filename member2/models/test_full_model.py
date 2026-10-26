import sys
from pathlib import Path

import torch

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


# --------------------------------------------------
# Import our Member 2 components
# --------------------------------------------------

from training.dummy_data import create_dummy_sequence
from training.augmentations import create_two_views

from models.gatv2_encoder import GATv2Encoder
from models.global_pooling import global_mean_pooling
from models.temporal_transformer import TemporalTransformer
from models.projection_head import ProjectionHead

from losses.nt_xent import NTXentLoss


# --------------------------------------------------
# 1. Create dummy temporal graph sequence
# --------------------------------------------------

sequence = create_dummy_sequence()

print("Original sequence")
print("Number of snapshots:", len(sequence))


# --------------------------------------------------
# 2. Create two stochastic views
# --------------------------------------------------

view1, view2 = create_two_views(
    sequence,
    dropout_probability=0.5
)

print("\nTwo views created")
print("View 1 snapshots:", len(view1))
print("View 2 snapshots:", len(view2))


# --------------------------------------------------
# 3. Create models
# --------------------------------------------------

gat_model = GATv2Encoder(
    in_channels=53,
    hidden_channels=128,
    heads=4
)

transformer = TemporalTransformer(
    input_dim=128,
    num_heads=4,
    num_layers=2
)

projection_head = ProjectionHead(
    input_dim=128,
    hidden_dim=128,
    output_dim=64
)

loss_function = NTXentLoss(
    temperature=0.1
)


# --------------------------------------------------
# 4. Function to encode one temporal sequence
# --------------------------------------------------

def encode_sequence(sequence):

    graph_vectors = []

    for graph in sequence:

        # ------------------------------------------
        # GATv2
        # ------------------------------------------

        node_embeddings = gat_model(
            graph.x,
            graph.edge_index
        )

        # ------------------------------------------
        # Global Mean Pooling
        # ------------------------------------------

        graph_embedding = global_mean_pooling(
            node_embeddings
        )

        graph_vectors.append(graph_embedding)

    # ----------------------------------------------
    # Combine 5 graph embeddings
    # ----------------------------------------------

    temporal_sequence = torch.cat(
        graph_vectors,
        dim=0
    )

    # [5,128] → [1,5,128]

    temporal_sequence = temporal_sequence.unsqueeze(0)

    # ----------------------------------------------
    # Temporal Transformer
    # ----------------------------------------------

    h = transformer(
        temporal_sequence
    )

    # ----------------------------------------------
    # Projection Head
    # ----------------------------------------------

    z = projection_head(
        h
    )

    return h, z


# --------------------------------------------------
# 5. Encode View 1
# --------------------------------------------------

h1, z1 = encode_sequence(view1)


# --------------------------------------------------
# 6. Encode View 2
# --------------------------------------------------

h2, z2 = encode_sequence(view2)


# --------------------------------------------------
# 7. Calculate NT-Xent loss
# --------------------------------------------------

loss = loss_function(
    z1,
    z2
)


# --------------------------------------------------
# 8. Print results
# --------------------------------------------------

print("\n======================================")
print("FULL MEMBER 2 PIPELINE")
print("======================================")

print("\nView 1:")
print("h1:", h1.shape)
print("z1:", z1.shape)

print("\nView 2:")
print("h2:", h2.shape)
print("z2:", z2.shape)

print("\nNT-Xent temperature:", loss_function.temperature)
print("NT-Xent loss:", loss.item())

print("\n======================================")
print("Pipeline completed successfully!")
print("======================================")