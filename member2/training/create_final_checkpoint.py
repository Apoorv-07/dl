import sys
import os
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


print("======================================")
print("MEMBER 2 DATASET 1 FINAL CHECKPOINT VALIDATION")
print("======================================")


# --------------------------------------------------
# CHECKPOINT PATH
# --------------------------------------------------

checkpoint_path = os.path.join(
    PROJECT_ROOT,
    "checkpoints_validation",
    "member2_dataset1_final_model.pt"
)

print()
print("Checkpoint:")
print(checkpoint_path)


# --------------------------------------------------
# LOAD CHECKPOINT IN FRESH PROCESS
# --------------------------------------------------

print()
print("Loading complete model object...")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

model = model.to(device)
model.eval()

print("Checkpoint loaded successfully.")

print()
print("Model type:")
print(type(model))


# --------------------------------------------------
# CHECK REQUIRED METHOD
# --------------------------------------------------

print()
print("======================================")
print("CHECKING MODEL CONTRACT")
print("======================================")

print()
print("Has encode_sequence:")
print(hasattr(model, "encode_sequence"))

if not hasattr(model, "encode_sequence"):
    raise RuntimeError(
        "ERROR: encode_sequence() is missing!"
    )

print()
print("encode_sequence() contract found.")


# --------------------------------------------------
# LOAD REAL DATA
# --------------------------------------------------

print()
print("======================================")
print("LOADING REAL DATA")
print("======================================")

graph_path = os.path.join(
    PROJECT_ROOT,
    "graph.pt"
)

data = torch.load(
    graph_path,
    map_location="cpu",
    weights_only=False
)

sequences = data["sequences"]

print()
print("Total sequences:")
print(len(sequences))


# --------------------------------------------------
# FIRST SEQUENCE
# --------------------------------------------------

sequence = sequences[0]

snapshots = sequence["snapshots"]

print()
print("Sequence ID:")
print(sequence["sequence_id"])

print()
print("Label:")
print(sequence["label"])

print()
print("Number of snapshots:")
print(len(snapshots))


# --------------------------------------------------
# MOVE SNAPSHOTS TO DEVICE
# --------------------------------------------------

snapshots = [
    snapshot.to(device)
    for snapshot in snapshots
]


# --------------------------------------------------
# TEST NORMAL MODE
# --------------------------------------------------

print()
print("======================================")
print("TESTING encode_sequence()")
print("======================================")

with torch.no_grad():

    output = model.encode_sequence(
        snapshots,
        return_attention=False
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
# VERIFY EMBEDDINGS
# --------------------------------------------------

assert "h" in output
assert "z" in output

assert output["h"].shape == (1, 128)
assert output["z"].shape == (1, 64)

print()
print("Embedding dimensions verified.")


# --------------------------------------------------
# TEST ATTENTION MODE
# --------------------------------------------------

print()
print("======================================")
print("TESTING ATTENTION MODE")
print("======================================")

with torch.no_grad():

    attention_output = model.encode_sequence(
        snapshots,
        return_attention=True
    )


print()
print("Returned keys:")
print(attention_output.keys())

assert "h" in attention_output
assert "z" in attention_output
assert "node_attention" in attention_output
assert "edge_attention" in attention_output


# --------------------------------------------------
# VERIFY ATTENTION
# --------------------------------------------------

node_attention = attention_output[
    "node_attention"
]

edge_attention = attention_output[
    "edge_attention"
]

print()
print("Node attention tensors:")
print(len(node_attention))

print()
print("Edge attention tensors:")
print(len(edge_attention))


assert len(node_attention) == len(snapshots)
assert len(edge_attention) == len(snapshots)


for i, snapshot in enumerate(snapshots):

    expected_nodes = snapshot.x.size(0)
    expected_edges = snapshot.edge_index.size(1)

    actual_nodes = node_attention[i].numel()
    actual_edges = edge_attention[i].numel()

    print()
    print(f"Snapshot {i + 1}")

    print(
        "Expected nodes:",
        expected_nodes
    )

    print(
        "Node attention:",
        actual_nodes
    )

    print(
        "Expected edges:",
        expected_edges
    )

    print(
        "Edge attention:",
        actual_edges
    )

    assert actual_nodes == expected_nodes
    assert actual_edges == expected_edges


# --------------------------------------------------
# FINAL RESULT
# --------------------------------------------------

print()
print("======================================")
print("FINAL VALIDATION RESULT")
print("======================================")

print()
print("SUCCESS!")

print()
print("Dataset 1 checkpoint can be loaded as a complete model.")

print()
print("encode_sequence() works.")

print()
print("Backbone h dimension: 128")

print("Projection z dimension: 64")

print()
print("Node attention dimensions: VALID")

print("Edge attention dimensions: VALID")

print()
print("======================================")
print("MEMBER 2 DATASET 1 CHECKPOINT VALIDATION COMPLETE")
print("======================================")