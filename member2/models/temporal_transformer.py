import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTransformer(nn.Module):

  def __init__(
      self, input_dim=128, num_heads=4, num_layers=2, dropout=0.1, max_len=5
  ):
    super().__init__()

    # Positional Embedding for 5 temporal snapshots
    self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, input_dim))
    nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=input_dim,
        nhead=num_heads,
        dim_feedforward=input_dim * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )

    self.transformer = nn.TransformerEncoder(
        encoder_layer, num_layers=num_layers
    )

    self.norm = nn.LayerNorm(input_dim)

  def forward(self, x):
    # x shape: [batch_size, 5, 128]

    # 1. Add positional encodings to preserve temporal snapshot order
    x = x + self.pos_embedding[:, : x.size(1), :]

    # 2. Pass through Temporal Transformer
    x = self.transformer(x)

    # 3. Mean across the 5 temporal snapshots
    h = x.mean(dim=1)

    # 4. LayerNorm + L2 normalize h to unit hypersphere
    h = self.norm(h)
    h = F.normalize(h, p=2, dim=-1)

    # h shape: [batch_size, 128]
    return h