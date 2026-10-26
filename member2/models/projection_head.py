import torch.nn as nn


class ProjectionHead(nn.Module):

    def __init__(
        self,
        input_dim=128,
        hidden_dim=128,
        output_dim=64
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, h):

        z = self.network(h)

        return z