import torch
from torch_geometric.nn import global_mean_pool


def global_mean_pooling(node_embeddings):
    """
    Convert node embeddings into one graph-level embedding.

    Input:
        node_embeddings: [num_nodes, hidden_dim]

    Output:
        graph_embedding: [1, hidden_dim]
    """

    # All nodes belong to the same graph
    batch = torch.zeros(
        node_embeddings.size(0),
        dtype=torch.long,
        device=node_embeddings.device
    )

    graph_embedding = global_mean_pool(
        node_embeddings,
        batch
    )

    return graph_embedding