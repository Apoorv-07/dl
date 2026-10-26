import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):

    def __init__(self, temperature=0.1):
        super().__init__()

        self.temperature = temperature

    def forward(self, z1, z2):

        # Normalize the embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Number of samples
        batch_size = z1.size(0)

        # Combine both views
        z = torch.cat([z1, z2], dim=0)

        # Cosine similarity matrix
        similarity = torch.matmul(z, z.T)

        # Scale by temperature
        similarity = similarity / self.temperature

        # Remove self-similarity
        mask = torch.eye(
            2 * batch_size,
            device=z.device,
            dtype=torch.bool
        )

        similarity.masked_fill_(mask, -float("inf"))

        # Positive pairs
        positives = torch.cat([
            torch.arange(
                batch_size,
                2 * batch_size,
                device=z.device
            ),
            torch.arange(
                0,
                batch_size,
                device=z.device
            )
        ])

        # Cross-entropy
        loss = F.cross_entropy(
            similarity,
            positives
        )

        return loss