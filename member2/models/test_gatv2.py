import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from training.dummy_data import create_dummy_sequence
from models.gatv2_encoder import GATv2Encoder


# Create dummy 5-snapshot sequence
sequence = create_dummy_sequence()


# Create GATv2 model
model = GATv2Encoder(
    in_channels=53,
    hidden_channels=128,
    heads=4
)


# Use the first graph snapshot
graph = sequence[0]


# Run the graph through GATv2
output = model(
    graph.x,
    graph.edge_index
)


print("\nInput shape:", graph.x.shape)
print("Output shape:", output.shape)