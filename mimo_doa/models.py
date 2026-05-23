"""PyTorch models: scenario classifier and covariance matrix reconstructor."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv1d(nn.Module):
    """Treat I/Q as two real channels and apply 1D convolution preserving channel split."""

    def __init__(self, in_complex: int, out_complex: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=2 * in_complex,
            out_channels=2 * out_complex,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ScenarioClassifier(nn.Module):
    """3-class classifier: single target / two targets / multipath.

    Input: real-valued tensor (B, 2, M, M) — sample covariance matrix split
    into real (channel 0) and imaginary (channel 1) parts.
    """

    def __init__(self, n_virtual: int = 8, n_classes: int = 3, hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(2, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden * 2)
        self.conv3 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden * 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(hidden * 2, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.bn1(self.conv1(x)))
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = torch.tanh(self.bn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CovReconstructor(nn.Module):
    """Reconstruct a clean covariance matrix from a noisy sample one.

    Input/Output shape: (B, 2, M, M) where channel 0 is real and channel 1 imag part.
    """

    def __init__(self, n_virtual: int = 8, hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(2, hidden, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.proj = nn.Conv2d(hidden, 2, kernel_size=1)
        self.fc = nn.Linear(2 * (n_virtual + 1) * (n_virtual + 1), 2 * n_virtual * n_virtual)
        self.n_virtual = n_virtual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.bn1(self.conv1(x)))
        h = torch.tanh(self.bn2(self.conv2(h)))
        h = self.proj(h)
        h = h.flatten(1)
        out = self.fc(h)
        return out.view(-1, 2, self.n_virtual, self.n_virtual)
