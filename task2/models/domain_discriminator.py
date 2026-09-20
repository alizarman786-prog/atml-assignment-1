"""Domain discriminator, shared architecture for DANN (Task 2 Step 3) and
CDAN (Task 2 Step 4). Same hidden width, activation, dropout for both, per
the assignment -- only the input dimension differs (512 for DANN's raw
feature; 512*num_classes for CDAN's multilinear feature-times-probability
map).
"""
from __future__ import annotations

import torch.nn as nn


class DomainDiscriminator(nn.Module):
    """256-unit hidden layer, ReLU, dropout 0.5, 2-class output (source vs.
    target), exactly as specified for both DANN and CDAN."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)
