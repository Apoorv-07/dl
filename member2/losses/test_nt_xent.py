import sys
from pathlib import Path

import torch


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from losses.nt_xent import NTXentLoss


# --------------------------------------------------
# 1. Create dummy embeddings
# --------------------------------------------------

batch_size = 8
embedding_dim = 64

z1 = torch.randn(
    batch_size,
    embedding_dim
)

z2 = torch.randn(
    batch_size,
    embedding_dim
)


# --------------------------------------------------
# 2. Create NT-Xent loss
# --------------------------------------------------

loss_function = NTXentLoss(
    temperature=0.1
)


# --------------------------------------------------
# 3. Calculate loss
# --------------------------------------------------

loss = loss_function(
    z1,
    z2
)


# --------------------------------------------------
# 4. Print results
# --------------------------------------------------

print("z1 shape:", z1.shape)
print("z2 shape:", z2.shape)

print("\nTemperature:", loss_function.temperature)

print("NT-Xent Loss:", loss.item())