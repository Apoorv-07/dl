import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.gatv2_encoder import GATv2Encoder
from models.global_pooling import global_mean_pooling
from models.temporal_transformer import TemporalTransformer
from models.projection_head import ProjectionHead
from training.augmentations import create_two_views
from losses.nt_xent import NTXentLoss


print("======================================")
print("REAL MULTI-SEQUENCE BATCH TEST")
print("======================================")

# --------------------------------------------------
# 1. Load real data
# --------------------------------------------------

data = torch.load(
    PROJECT_ROOT / "graph.pt",
    weights_only=False
)

sequences = data["sequences"]

print("\nTotal sequences:", len(sequences))

# --------------------------------------------------
# 2. Select 4 real sequences
# --------------------------------------------------

batch_size = 4

selected_sequences = sequences[:batch_size]

print("Selected sequences:", len(selected_sequences))

# --------------------------------------------------
# 3. Create two views for every sequence
# --------------------------------------------------

view1_batch = []
view2_batch = []

for sequence in selected_sequences:

    snapshots = sequence["snapshots"]

    view1, view2 = create_two_views(
        snapshots,
        dropout_probability=0.5
    )

    view1_batch.append(view1)
    view2_batch.append(view2)

print("View 1 sequences:", len(view1_batch))
print("View 2 sequences:", len(view2_batch))

# --------------------------------------------------
# 4. Models
# --------------------------------------------------

gat_model = GATv2Encoder(
    in_channels=53,
    hidden_channels=128,
    heads=4,
    dropout=0.2
)

transformer = TemporalTransformer(
    input_dim=128,
    num_heads=4,
    num_layers=2,
    dropout=0.1
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
# 5. Encode one temporal sequence
# --------------------------------------------------

def encode_sequence(sequence):

    graph_embeddings = []

    for graph in sequence:

        node_embeddings = gat_model(
            graph.x,
            graph.edge_index,
            graph.edge_attr
        )

        graph_embedding = global_mean_pooling(
            node_embeddings
        )

        graph_embeddings.append(graph_embedding)

    temporal_sequence = torch.cat(
        graph_embeddings,
        dim=0
    )

    temporal_sequence = temporal_sequence.unsqueeze(0)

    h = transformer(
        temporal_sequence
    )

    z = projection_head(
        h
    )

    return h, z


# --------------------------------------------------
# 6. Encode all sequences
# --------------------------------------------------

h1_batch = []
z1_batch = []

h2_batch = []
z2_batch = []

print("\n======================================")
print("ENCODING BATCH")
print("======================================")

with torch.no_grad():

    for i in range(batch_size):

        print(f"\nSequence {i + 1}")

        h1, z1 = encode_sequence(
            view1_batch[i]
        )

        h2, z2 = encode_sequence(
            view2_batch[i]
        )

        print("h1:", h1.shape)
        print("z1:", z1.shape)

        print("h2:", h2.shape)
        print("z2:", z2.shape)

        h1_batch.append(h1)
        z1_batch.append(z1)

        h2_batch.append(h2)
        z2_batch.append(z2)

# --------------------------------------------------
# 7. Combine into batches
# --------------------------------------------------

h1_batch = torch.cat(
    h1_batch,
    dim=0
)

z1_batch = torch.cat(
    z1_batch,
    dim=0
)

h2_batch = torch.cat(
    h2_batch,
    dim=0
)

z2_batch = torch.cat(
    z2_batch,
    dim=0
)

# --------------------------------------------------
# 8. NT-Xent
# --------------------------------------------------

loss = loss_function(
    z1_batch,
    z2_batch
)

# --------------------------------------------------
# 9. Results
# --------------------------------------------------

print("\n======================================")
print("BATCH RESULTS")
print("======================================")

print("h1 batch:", h1_batch.shape)
print("h2 batch:", h2_batch.shape)

print("z1 batch:", z1_batch.shape)
print("z2 batch:", z2_batch.shape)

print("\nNT-Xent temperature:")
print(loss_function.temperature)

print("\nNT-Xent loss:")
print(loss.item())

print("\n======================================")
print("TEST RESULT")
print("======================================")

if (
    h1_batch.shape == torch.Size([4, 128])
    and
    h2_batch.shape == torch.Size([4, 128])
    and
    z1_batch.shape == torch.Size([4, 64])
    and
    z2_batch.shape == torch.Size([4, 64])
):

    print("SUCCESS: Multi-sequence batch created correctly.")

else:

    print("WARNING: Unexpected batch dimensions.")

print("======================================")