"""Signal-conditioned dual-network model skeletons.

These modules are intentionally limited to model definitions. They do not
load data, run optimization, or save checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _init_linear(layer: nn.Linear) -> nn.Linear:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def _init_conv1d(layer: nn.Conv1d) -> nn.Conv1d:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def _build_tanh_mlp(input_dim: int, output_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")

    layers = []
    in_dim = input_dim
    for _ in range(num_layers):
        layer = _init_linear(nn.Linear(in_dim, hidden_dim))
        layers.append(layer)
        layers.append(nn.Tanh())
        in_dim = hidden_dim
    output_layer = _init_linear(nn.Linear(in_dim, output_dim))
    layers.append(output_layer)
    return nn.Sequential(*layers)


class BzEncoder(nn.Module):
    """Encode a measured Bz signal into one latent vector per sample."""

    def __init__(self, signal_len: int, hidden_dim: int = 128, latent_dim: int = 64, num_layers: int = 2):
        super().__init__()
        if signal_len <= 0:
            raise ValueError("signal_len must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.signal_len = int(signal_len)
        self.latent_dim = int(latent_dim)
        self.net = _build_tanh_mlp(
            input_dim=self.signal_len,
            output_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        if signals.ndim != 2:
            raise ValueError(f"signals must have shape [B, signal_len], got {tuple(signals.shape)}")
        if signals.shape[1] != self.signal_len:
            raise ValueError(
                f"signals second dimension must be {self.signal_len}, got {signals.shape[1]}"
            )
        return self.net(signals)


class ConvBzEncoder(nn.Module):
    """Encode a measured Bz signal with a small 1D CNN."""

    def __init__(self, signal_len: int, hidden_dim: int = 128, latent_dim: int = 64, num_layers: int = 2):
        super().__init__()
        if signal_len <= 0:
            raise ValueError("signal_len must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.signal_len = int(signal_len)
        self.latent_dim = int(latent_dim)

        conv_layers = []
        in_channels = 1
        for _ in range(num_layers):
            conv_layers.append(_init_conv1d(nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2)))
            conv_layers.append(nn.Tanh())
            in_channels = hidden_dim
        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*conv_layers)
        self.proj = _init_linear(nn.Linear(hidden_dim, self.latent_dim))

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        if signals.ndim != 2:
            raise ValueError(f"signals must have shape [B, signal_len], got {tuple(signals.shape)}")
        if signals.shape[1] != self.signal_len:
            raise ValueError(
                f"signals second dimension must be {self.signal_len}, got {signals.shape[1]}"
            )
        features = self.conv(signals.unsqueeze(1)).squeeze(-1)
        return self.proj(features)


class ConditionalMLP(nn.Module):
    """Coordinate MLP conditioned on a per-sample latent vector."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        conditioning_mode: str = "concat",
        extra_point_dim: int = 0,
    ):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if conditioning_mode not in {"concat", "film"}:
            raise ValueError("conditioning_mode must be 'concat' or 'film'")
        if extra_point_dim < 0:
            raise ValueError("extra_point_dim must be non-negative")
        self.latent_dim = int(latent_dim)
        self.extra_point_dim = int(extra_point_dim)
        self.conditioning_mode = conditioning_mode
        if self.conditioning_mode == "concat":
            self.net = _build_tanh_mlp(
                input_dim=2 + self.latent_dim + self.extra_point_dim,
                output_dim=1,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        else:
            self.coord_layers = nn.ModuleList()
            self.film_layers = nn.ModuleList()
            in_dim = 2 + self.extra_point_dim
            for _ in range(num_layers):
                self.coord_layers.append(_init_linear(nn.Linear(in_dim, hidden_dim)))
                self.film_layers.append(_init_linear(nn.Linear(self.latent_dim, 2 * hidden_dim)))
                in_dim = hidden_dim
            self.output_layer = _init_linear(nn.Linear(hidden_dim, 1))

    def _prepare_inputs(
        self,
        coords: torch.Tensor,
        latent: torch.Tensor,
        point_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if latent.ndim != 2:
            raise ValueError(f"latent must have shape [B, latent_dim], got {tuple(latent.shape)}")
        if latent.shape[1] != self.latent_dim:
            raise ValueError(f"latent second dimension must be {self.latent_dim}, got {latent.shape[1]}")

        batch_size = latent.shape[0]
        if coords.ndim == 2:
            if coords.shape[1] != 2:
                raise ValueError(f"coords with rank 2 must have shape [N, 2], got {tuple(coords.shape)}")
            coords_batched = coords.unsqueeze(0).expand(batch_size, -1, -1)
        elif coords.ndim == 3:
            if coords.shape[2] != 2:
                raise ValueError(f"coords with rank 3 must have shape [B, N, 2], got {tuple(coords.shape)}")
            if coords.shape[0] != batch_size:
                raise ValueError(
                    f"coords batch dimension must match latent batch {batch_size}, got {coords.shape[0]}"
                )
            coords_batched = coords
        else:
            raise ValueError(f"coords must have shape [N, 2] or [B, N, 2], got {tuple(coords.shape)}")

        point_features_batched = None
        if self.extra_point_dim == 0:
            if point_features is not None:
                raise ValueError("point_features were provided but extra_point_dim is 0")
        else:
            if point_features is None:
                raise ValueError("point_features are required when extra_point_dim is positive")
            if point_features.ndim != 3:
                raise ValueError(
                    f"point_features must have shape [B, N, K], got {tuple(point_features.shape)}"
                )
            if point_features.shape[0] != batch_size:
                raise ValueError(
                    f"point_features batch dimension must match latent batch {batch_size}, got {point_features.shape[0]}"
                )
            if point_features.shape[1] != coords_batched.shape[1]:
                raise ValueError(
                    "point_features point dimension must match coords point dimension "
                    f"{coords_batched.shape[1]}, got {point_features.shape[1]}"
                )
            if point_features.shape[2] != self.extra_point_dim:
                raise ValueError(
                    f"point_features feature dimension must be {self.extra_point_dim}, got {point_features.shape[2]}"
                )
            point_features_batched = point_features

        latent_batched = latent.unsqueeze(1).expand(-1, coords_batched.shape[1], -1)
        return coords_batched, latent_batched, point_features_batched

    def forward(
        self,
        coords: torch.Tensor,
        latent: torch.Tensor,
        point_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        coords_batched, latent_batched, point_features_batched = self._prepare_inputs(
            coords=coords,
            latent=latent,
            point_features=point_features,
        )
        coord_features = coords_batched
        if point_features_batched is not None:
            coord_features = torch.cat([coord_features, point_features_batched], dim=-1)
        if self.conditioning_mode == "concat":
            return self.net(torch.cat([coord_features, latent_batched], dim=-1))

        hidden = coord_features
        for coord_layer, film_layer in zip(self.coord_layers, self.film_layers):
            hidden = coord_layer(hidden)
            gamma_beta = film_layer(latent_batched)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            hidden = hidden * (1.0 + gamma) + beta
            hidden = torch.tanh(hidden)
        return self.output_layer(hidden)


class ConditionalMuNet(nn.Module):
    """Signal-conditioned bounded permeability network."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        mu_min: float = 1.0,
        mu_max: float = 1000.0,
        conditioning_mode: str = "concat",
        extra_point_dim: int = 0,
    ):
        super().__init__()
        if mu_max <= mu_min:
            raise ValueError("mu_max must be greater than mu_min")
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.raw_net = ConditionalMLP(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            conditioning_mode=conditioning_mode,
            extra_point_dim=extra_point_dim,
        )

    def forward(
        self,
        coords: torch.Tensor,
        latent: torch.Tensor,
        point_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw_mu = self.raw_net(coords=coords, latent=latent, point_features=point_features)
        mu01 = torch.sigmoid(raw_mu)
        return self.mu_min + (self.mu_max - self.mu_min) * mu01


class ConditionalPhiNet(nn.Module):
    """Signal-conditioned scalar-potential network."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        conditioning_mode: str = "concat",
        extra_point_dim: int = 0,
    ):
        super().__init__()
        self.net = ConditionalMLP(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            conditioning_mode=conditioning_mode,
            extra_point_dim=extra_point_dim,
        )

    def forward(
        self,
        coords: torch.Tensor,
        latent: torch.Tensor,
        point_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.net(coords=coords, latent=latent, point_features=point_features)


class ConditionalMaskNet(nn.Module):
    """Signal-conditioned direct defect-mask logit network."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        conditioning_mode: str = "concat",
        extra_point_dim: int = 0,
    ):
        super().__init__()
        self.net = ConditionalMLP(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            conditioning_mode=conditioning_mode,
            extra_point_dim=extra_point_dim,
        )

    def forward(
        self,
        coords: torch.Tensor,
        latent: torch.Tensor,
        point_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.net(coords=coords, latent=latent, point_features=point_features)


class ConditionalDualNet(nn.Module):
    """Encode Bz signals and predict conditional mu / phi fields at coords."""

    def __init__(
        self,
        signal_len: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        mu_min: float = 1.0,
        mu_max: float = 1000.0,
        conditioning_mode: str = "concat",
        encoder_type: str = "mlp",
        extra_point_dim: int = 0,
        predict_mask: bool = False,
    ):
        super().__init__()
        if conditioning_mode not in {"concat", "film"}:
            raise ValueError("conditioning_mode must be 'concat' or 'film'")
        if encoder_type not in {"mlp", "cnn"}:
            raise ValueError("encoder_type must be 'mlp' or 'cnn'")
        if extra_point_dim < 0:
            raise ValueError("extra_point_dim must be non-negative")
        self.predict_mask = bool(predict_mask)
        encoder_cls = BzEncoder if encoder_type == "mlp" else ConvBzEncoder
        self.encoder = encoder_cls(
            signal_len=signal_len,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        self.mu_net = ConditionalMuNet(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mu_min=mu_min,
            mu_max=mu_max,
            conditioning_mode=conditioning_mode,
            extra_point_dim=extra_point_dim,
        )
        self.mask_net = None
        if self.predict_mask:
            self.mask_net = ConditionalMaskNet(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                conditioning_mode=conditioning_mode,
                extra_point_dim=extra_point_dim,
            )
        self.phi_net = ConditionalPhiNet(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            conditioning_mode=conditioning_mode,
            extra_point_dim=extra_point_dim,
        )

    def forward(
        self,
        signals: torch.Tensor,
        coords: torch.Tensor,
        point_features: torch.Tensor | None = None,
        return_phi: bool = True,
    ) -> dict[str, torch.Tensor]:
        latent = self.encoder(signals)
        mu = self.mu_net(coords=coords, latent=latent, point_features=point_features)
        result = {
            "latent": latent,
            "mu": mu,
        }
        if return_phi:
            result["phi"] = self.phi_net(coords=coords, latent=latent, point_features=point_features)
        if self.mask_net is not None:
            mask_logits = self.mask_net(coords=coords, latent=latent, point_features=point_features)
            result["mask_logits"] = mask_logits
            result["mask_prob"] = torch.sigmoid(mask_logits)
        return result
