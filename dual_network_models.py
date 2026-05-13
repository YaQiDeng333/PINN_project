"""Minimal model definitions for the dual-network variational branch."""

import torch
import torch.nn as nn


def _build_tanh_mlp(input_dim, output_dim, hidden_dim, num_layers):
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")

    layers = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())
        in_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class PhiNet(nn.Module):
    """Coordinate MLP for the scalar potential phi(x, y)."""

    def __init__(self, input_dim=2, hidden_dim=128, num_layers=4):
        super().__init__()
        self.net = _build_tanh_mlp(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, coords):
        return self.net(coords)


class MuNet(nn.Module):
    """Coordinate MLP for bounded permeability mu(x, y)."""

    def __init__(
        self,
        input_dim=2,
        hidden_dim=128,
        num_layers=4,
        mu_min=1.0,
        mu_max=1000.0,
    ):
        super().__init__()
        if mu_max <= mu_min:
            raise ValueError("mu_max must be greater than mu_min")
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.net = _build_tanh_mlp(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, coords):
        raw_mu = self.net(coords)
        mu01 = torch.sigmoid(raw_mu)
        return self.mu_min + (self.mu_max - self.mu_min) * mu01
