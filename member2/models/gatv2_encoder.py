import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GATv2Encoder(nn.Module):

    def __init__(
        self,
        in_channels=53,
        hidden_channels=128,
        heads=4,
        dropout=0.2
    ):
        super().__init__()

        # First GATv2 layer
        self.gat1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=in_channels
        )

        # Second GATv2 layer
        self.gat2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,
            dropout=dropout,
            edge_dim=in_channels
        )

        self.activation = nn.ELU()

    def forward(self, x, edge_index, edge_attr):

        # GATv2 Layer 1
        x = self.gat1(
            x,
            edge_index,
            edge_attr
        )

        x = self.activation(x)

        # GATv2 Layer 2
        x = self.gat2(
            x,
            edge_index,
            edge_attr
        )

        return x