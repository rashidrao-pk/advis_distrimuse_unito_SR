from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=pad,
        )

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.chunk(self.gates(torch.cat([x, h], dim=1)), 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch, height, width, device, dtype):
        shape = (batch, self.hidden_channels, height, width)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )


class SpatialRiskConvLSTM(nn.Module):
    """Shared per-area ConvLSTM forecaster for multiple future horizons."""

    def __init__(self, hidden_channels: int = 32, horizons: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.rnn = ConvLSTMCell(16, hidden_channels)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, horizons, 1),
        )

    def forward(self, x):
        # x: [B,T,1,H,W]
        b, t, _, h, w = x.shape
        state = self.rnn.init_state(b, h, w, x.device, x.dtype)
        for step in range(t):
            z = self.encoder(x[:, step])
            state = self.rnn(z, state)
        logits = self.head(state[0])          # [B,K,H,W]
        return torch.sigmoid(logits).unsqueeze(2)  # [B,K,1,H,W]
