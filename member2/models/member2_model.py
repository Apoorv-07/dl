import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from models.temporal_transformer import TemporalTransformer
from models.projection_head import ProjectionHead


class Member2FoundationModel(nn.Module):

    def __init__(
        self,
        in_channels=53,
        hidden_channels=128,
        heads=4,
        transformer_heads=4,
        transformer_layers=2,
        h_dim=128,
        z_dim=64,
        dropout=0.2,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # GATv2 encoder
        # SAME structure used during training
        # ---------------------------------------------------------

        self.gat1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=in_channels,
        )

        self.gat2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,
            dropout=dropout,
            edge_dim=in_channels,
        )

        self.activation = nn.ELU()

        # ---------------------------------------------------------
        # Temporal Transformer
        # IMPORTANT:
        # Use the exact class that was used during training.
        # Training dropout = 0.1
        # ---------------------------------------------------------

        self.temporal_transformer = TemporalTransformer(
            input_dim=h_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=0.1,
        )

        # ---------------------------------------------------------
        # Projection head
        # IMPORTANT:
        # Use the exact ProjectionHead used during training.
        # It uses GELU.
        # ---------------------------------------------------------

        self.projection_head = ProjectionHead(
            input_dim=h_dim,
            hidden_dim=h_dim,
            output_dim=z_dim,
        )

    # =========================================================
    # Encode one graph snapshot
    # =========================================================

    def encode_snapshot(
        self,
        snapshot,
        return_attention=False,
    ):

        x = snapshot.x
        edge_index = snapshot.edge_index
        edge_attr = snapshot.edge_attr

        # -----------------------------------------------------
        # First GATv2 layer
        # -----------------------------------------------------

        if return_attention:

            x, (attention_edge_index, attention_weights) = self.gat1(
                x,
                edge_index,
                edge_attr,
                return_attention_weights=True,
            )

        else:

            x = self.gat1(
                x,
                edge_index,
                edge_attr,
            )

            attention_edge_index = None
            attention_weights = None

        x = self.activation(x)

        # -----------------------------------------------------
        # Second GATv2 layer
        # -----------------------------------------------------

        x = self.gat2(
            x,
            edge_index,
            edge_attr,
        )

        # -----------------------------------------------------
        # Global mean pooling
        # -----------------------------------------------------

        graph_embedding = x.mean(
            dim=0,
            keepdim=True,
        )

        # -----------------------------------------------------
        # No attention requested
        # -----------------------------------------------------

        if not return_attention:

            return graph_embedding

        # =====================================================
        # Process attention
        # =====================================================

        # Average across GAT heads
        edge_attention = attention_weights.mean(
            dim=-1
        )

        num_original_edges = edge_index.size(1)

        # -----------------------------------------------------
        # Align GAT attention with original graph edges
        # GATv2 may add self-loops internally.
        # -----------------------------------------------------

        aligned_edge_attention = torch.zeros(
            num_original_edges,
            device=edge_attention.device,
            dtype=edge_attention.dtype,
        )

        attention_lookup = {}

        for i in range(
            attention_edge_index.size(1)
        ):

            src = int(
                attention_edge_index[0, i]
            )

            dst = int(
                attention_edge_index[1, i]
            )

            if (src, dst) not in attention_lookup:

                attention_lookup[
                    (src, dst)
                ] = i

        for i in range(
            num_original_edges
        ):

            src = int(
                edge_index[0, i]
            )

            dst = int(
                edge_index[1, i]
            )

            index = attention_lookup.get(
                (src, dst)
            )

            if index is not None:

                aligned_edge_attention[i] = (
                    edge_attention[index]
                )

        # -----------------------------------------------------
        # Normalize edge attention
        # -----------------------------------------------------

        aligned_edge_attention = torch.clamp(
            aligned_edge_attention,
            min=0.0,
        )

        edge_sum = (
            aligned_edge_attention.sum()
        )

        if edge_sum > 0:

            aligned_edge_attention = (
                aligned_edge_attention
                / edge_sum
            )

        elif num_original_edges > 0:

            aligned_edge_attention = torch.full(
                (num_original_edges,),
                1.0 / num_original_edges,
                device=aligned_edge_attention.device,
                dtype=aligned_edge_attention.dtype,
            )

        # -----------------------------------------------------
        # Convert edge attention to node attention
        # -----------------------------------------------------

        num_nodes = snapshot.num_nodes

        node_attention = torch.zeros(
            num_nodes,
            device=edge_attention.device,
            dtype=edge_attention.dtype,
        )

        if num_original_edges > 0:

            destination_nodes = edge_index[1]

            node_attention.index_add_(
                0,
                destination_nodes,
                aligned_edge_attention,
            )

        # -----------------------------------------------------
        # Normalize node attention
        # -----------------------------------------------------

        node_sum = node_attention.sum()

        if node_sum > 0:

            node_attention = (
                node_attention / node_sum
            )

        elif num_nodes > 0:

            node_attention = torch.full(
                (num_nodes,),
                1.0 / num_nodes,
                device=node_attention.device,
                dtype=node_attention.dtype,
            )

        return (
            graph_embedding,
            node_attention,
            aligned_edge_attention,
        )

    # =========================================================
    # Encode complete temporal sequence
    # =========================================================

    def encode_sequence(
        self,
        snapshots,
        return_attention=False,
    ):

        temporal_embeddings = []

        node_attention_list = []
        edge_attention_list = []

        # -----------------------------------------------------
        # Encode all snapshots
        # -----------------------------------------------------

        for snapshot in snapshots:

            if return_attention:

                (
                    graph_embedding,
                    node_attention,
                    edge_attention,
                ) = self.encode_snapshot(
                    snapshot,
                    return_attention=True,
                )

                temporal_embeddings.append(
                    graph_embedding
                )

                node_attention_list.append(
                    node_attention.detach().cpu()
                )

                edge_attention_list.append(
                    edge_attention.detach().cpu()
                )

            else:

                graph_embedding = (
                    self.encode_snapshot(
                        snapshot,
                        return_attention=False,
                    )
                )

                temporal_embeddings.append(
                    graph_embedding
                )

        # -----------------------------------------------------
        # Shape:
        # [1, number_of_snapshots, 128]
        # -----------------------------------------------------

        temporal_input = torch.stack(
            temporal_embeddings,
            dim=1,
        )

        # -----------------------------------------------------
        # EXACT SAME temporal operation as training
        #
        # TemporalTransformer.forward():
        #
        # x = transformer(x)
        # h = x.mean(dim=1)
        #
        # -----------------------------------------------------

        h = self.temporal_transformer(
            temporal_input
        )

        # -----------------------------------------------------
        # EXACT SAME projection head as training
        # -----------------------------------------------------

        z = self.projection_head(h)

        result = {
            "h": h,
            "z": z,
        }

        if return_attention:

            result["node_attention"] = (
                node_attention_list
            )

            result["edge_attention"] = (
                edge_attention_list
            )

        return result

    # =========================================================
    # Simple encode interface
    # =========================================================

    def encode(
        self,
        graph_sequence,
    ):

        output = self.encode_sequence(
            graph_sequence["snapshots"],
            return_attention=False,
        )

        return output["h"]

    # =========================================================
    # Metadata
    # =========================================================

    def metadata(self):

        return {
            "model_name": "Member2FoundationModel",
            "input_features": 53,
            "hidden_dimension": 128,
            "embedding_dimension_h": 128,
            "projection_dimension_z": 64,
            "gat_heads": 4,
            "transformer_heads": 4,
            "transformer_layers": 2,
            "temporal_snapshots": 5,
            "training_embedding": "h",
            "contrastive_embedding": "z",
            "projection_activation": "GELU",
        }