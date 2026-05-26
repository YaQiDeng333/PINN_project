"""PyTorch modules for COMSOL parametric inverse experiments."""

from __future__ import annotations

import torch
from torch import nn


class ParametricBzEncoder(nn.Module):
    """MLP encoder for flattened multi-height Bz signals."""

    def __init__(
        self,
        signal_len: int = 600,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if signal_len <= 0:
            raise ValueError("signal_len must be positive.")
        if hidden_dim <= 0 or latent_dim <= 0:
            raise ValueError("hidden_dim and latent_dim must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self.signal_len = signal_len
        layers: list[nn.Module] = []
        in_dim = signal_len
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        if signals.ndim != 2:
            raise ValueError(f"signals must have shape [B, signal_len], got {tuple(signals.shape)}")
        if signals.shape[1] != self.signal_len:
            raise ValueError(f"Expected signal_len={self.signal_len}, got {signals.shape[1]}")
        return self.net(signals)


class Conv1dBzEncoder(nn.Module):
    """1D CNN encoder for flattened Bz signals."""

    def __init__(
        self,
        signal_len: int = 600,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        attention: bool = False,
    ) -> None:
        super().__init__()
        if signal_len <= 0:
            raise ValueError("signal_len must be positive.")
        if hidden_dim <= 0 or latent_dim <= 0:
            raise ValueError("hidden_dim and latent_dim must be positive.")
        self.signal_len = signal_len
        self.attention = attention
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2 if hidden_dim >= 2 else 1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2 if hidden_dim >= 2 else 1, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.attention_score = nn.Linear(hidden_dim, 1) if attention else None
        self.project = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.ReLU())

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        if signals.ndim != 2:
            raise ValueError(f"signals must have shape [B, signal_len], got {tuple(signals.shape)}")
        if signals.shape[1] != self.signal_len:
            raise ValueError(f"Expected signal_len={self.signal_len}, got {signals.shape[1]}")
        features = self.conv(signals.unsqueeze(1)).transpose(1, 2)
        if self.attention_score is None:
            pooled = features.mean(dim=1)
        else:
            weights = torch.softmax(self.attention_score(features), dim=1)
            pooled = (features * weights).sum(dim=1)
        return self.project(pooled)


class FeatureMLP(nn.Module):
    """MLP encoder for handcrafted physics features."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, latent_dim: int = 64) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if hidden_dim <= 0 or latent_dim <= 0:
            raise ValueError("hidden_dim and latent_dim must be positive.")
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must have shape [B, feature_dim], got {tuple(features.shape)}")
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {features.shape[1]}")
        return self.net(features)


class ComponentParamHead(nn.Module):
    """Predict component presence, type and continuous geometry parameters."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        max_components: int = 3,
        num_types: int = 2,
        num_continuous: int = 6,
        head_mode: str = "shared",
        center_representation: str = "continuous",
        center_x_bins: int = 0,
        center_y_bins: int = 0,
    ) -> None:
        super().__init__()
        if max_components <= 0:
            raise ValueError("max_components must be positive.")
        if num_types <= 0:
            raise ValueError("num_types must be positive.")
        if num_continuous <= 0:
            raise ValueError("num_continuous must be positive.")
        if head_mode not in {"shared", "component_specific"}:
            raise ValueError("head_mode must be 'shared' or 'component_specific'.")
        if center_representation not in {"continuous", "bin_offset"}:
            raise ValueError("center_representation must be 'continuous' or 'bin_offset'.")
        if center_representation == "bin_offset" and (center_x_bins <= 0 or center_y_bins <= 0):
            raise ValueError("center_x_bins and center_y_bins must be positive for bin_offset centers.")
        self.max_components = max_components
        self.num_types = num_types
        self.num_continuous = num_continuous
        self.head_mode = head_mode
        self.center_representation = center_representation
        self.center_x_bins = center_x_bins
        self.center_y_bins = center_y_bins
        if head_mode == "shared":
            self.shared = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
            self.presence = nn.Linear(hidden_dim, max_components)
            self.type_logits = nn.Linear(hidden_dim, max_components * num_types)
            self.continuous = nn.Linear(hidden_dim, max_components * num_continuous)
            if center_representation == "bin_offset":
                self.center_x_bin_logits = nn.Linear(hidden_dim, max_components * center_x_bins)
                self.center_y_bin_logits = nn.Linear(hidden_dim, max_components * center_y_bins)
                self.center_offset = nn.Linear(hidden_dim, max_components * 2)
        else:
            self.component_heads = nn.ModuleList(
                [
                    self._build_component_head(latent_dim, hidden_dim)
                    for _ in range(max_components)
                ]
            )

    def _build_component_head(self, latent_dim: int, hidden_dim: int) -> nn.ModuleDict:
        head = nn.ModuleDict(
            {
                "shared": nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU()),
                "presence": nn.Linear(hidden_dim, 1),
                "type_logits": nn.Linear(hidden_dim, self.num_types),
                "continuous": nn.Linear(hidden_dim, self.num_continuous),
            }
        )
        if self.center_representation == "bin_offset":
            head["center_x_bin_logits"] = nn.Linear(hidden_dim, self.center_x_bins)
            head["center_y_bin_logits"] = nn.Linear(hidden_dim, self.center_y_bins)
            head["center_offset"] = nn.Linear(hidden_dim, 2)
        return head

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent.ndim != 2:
            raise ValueError(f"latent must have shape [B, latent_dim], got {tuple(latent.shape)}")
        batch = latent.shape[0]
        if self.head_mode == "shared":
            h = self.shared(latent)
            presence_logits = self.presence(h)
            type_logits = self.type_logits(h).view(batch, self.max_components, self.num_types)
            continuous = self.continuous(h).view(batch, self.max_components, self.num_continuous)
            if self.center_representation == "bin_offset":
                center_x_bin_logits = self.center_x_bin_logits(h).view(batch, self.max_components, self.center_x_bins)
                center_y_bin_logits = self.center_y_bin_logits(h).view(batch, self.max_components, self.center_y_bins)
                center_offset = self.center_offset(h).view(batch, self.max_components, 2)
        else:
            presence_parts = []
            type_parts = []
            continuous_parts = []
            center_x_bin_parts = []
            center_y_bin_parts = []
            center_offset_parts = []
            for head in self.component_heads:
                h = head["shared"](latent)
                presence_parts.append(head["presence"](h))
                type_parts.append(head["type_logits"](h).unsqueeze(1))
                continuous_parts.append(head["continuous"](h).unsqueeze(1))
                if self.center_representation == "bin_offset":
                    center_x_bin_parts.append(head["center_x_bin_logits"](h).unsqueeze(1))
                    center_y_bin_parts.append(head["center_y_bin_logits"](h).unsqueeze(1))
                    center_offset_parts.append(head["center_offset"](h).unsqueeze(1))
            presence_logits = torch.cat(presence_parts, dim=1)
            type_logits = torch.cat(type_parts, dim=1)
            continuous = torch.cat(continuous_parts, dim=1)
            if self.center_representation == "bin_offset":
                center_x_bin_logits = torch.cat(center_x_bin_parts, dim=1)
                center_y_bin_logits = torch.cat(center_y_bin_parts, dim=1)
                center_offset = torch.cat(center_offset_parts, dim=1)
        out = {
            "presence_logits": presence_logits,
            "presence_prob": torch.sigmoid(presence_logits),
            "type_logits": type_logits,
            "continuous": continuous,
        }
        if self.center_representation == "bin_offset":
            out.update(
                {
                    "center_x_bin_logits": center_x_bin_logits,
                    "center_y_bin_logits": center_y_bin_logits,
                    "center_offset": center_offset,
                }
            )
        return out


class AuxCenterHead(nn.Module):
    """Auxiliary center-bin head attached to the inverse latent."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        max_components: int = 3,
        center_x_bins: int = 0,
        center_y_bins: int = 0,
    ) -> None:
        super().__init__()
        if max_components <= 0:
            raise ValueError("max_components must be positive.")
        if center_x_bins <= 0 or center_y_bins <= 0:
            raise ValueError("center_x_bins and center_y_bins must be positive for auxiliary center head.")
        self.max_components = max_components
        self.center_x_bins = center_x_bins
        self.center_y_bins = center_y_bins
        self.shared = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.center_x_bin_logits = nn.Linear(hidden_dim, max_components * center_x_bins)
        self.center_y_bin_logits = nn.Linear(hidden_dim, max_components * center_y_bins)
        self.center_offset = nn.Linear(hidden_dim, max_components * 2)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent.ndim != 2:
            raise ValueError(f"latent must have shape [B, latent_dim], got {tuple(latent.shape)}")
        batch = latent.shape[0]
        h = self.shared(latent)
        return {
            "aux_center_x_bin_logits": self.center_x_bin_logits(h).view(batch, self.max_components, self.center_x_bins),
            "aux_center_y_bin_logits": self.center_y_bin_logits(h).view(batch, self.max_components, self.center_y_bins),
            "aux_center_offset": self.center_offset(h).view(batch, self.max_components, 2),
        }


class ParametricInverseNet(nn.Module):
    """Predict component-level geometry targets from flattened Bz signals."""

    def __init__(
        self,
        signal_len: int = 600,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        max_components: int = 3,
        num_types: int = 2,
        num_continuous: int = 6,
        num_layers: int = 3,
        encoder_type: str = "mlp",
        head_mode: str = "shared",
        feature_dim: int = 0,
        feature_fusion_mode: str = "none",
        center_representation: str = "continuous",
        center_x_bins: int = 0,
        center_y_bins: int = 0,
        aux_center_head: bool = False,
    ) -> None:
        super().__init__()
        if feature_fusion_mode not in {"none", "features_only", "concat_latent"}:
            raise ValueError("feature_fusion_mode must be 'none', 'features_only', or 'concat_latent'.")
        if feature_fusion_mode != "none" and feature_dim <= 0:
            raise ValueError("feature_dim must be positive when feature_fusion_mode is enabled.")
        self.feature_fusion_mode = feature_fusion_mode
        self.feature_dim = feature_dim
        self.encoder = None
        if feature_fusion_mode == "features_only":
            self.feature_encoder = FeatureMLP(feature_dim=feature_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
            self.fusion = None
        elif encoder_type == "mlp":
            self.encoder = ParametricBzEncoder(
                signal_len=signal_len,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                num_layers=num_layers,
            )
        elif encoder_type == "cnn1d":
            self.encoder = Conv1dBzEncoder(
                signal_len=signal_len,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                attention=False,
            )
        elif encoder_type == "cnn1d_attention":
            self.encoder = Conv1dBzEncoder(
                signal_len=signal_len,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                attention=True,
            )
        else:
            raise ValueError("encoder_type must be 'mlp', 'cnn1d', or 'cnn1d_attention'.")
        if feature_fusion_mode == "concat_latent":
            self.feature_encoder = FeatureMLP(feature_dim=feature_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
            self.fusion = nn.Sequential(nn.Linear(2 * latent_dim, latent_dim), nn.ReLU())
        elif feature_fusion_mode == "none":
            self.feature_encoder = None
            self.fusion = None
        self.head = ComponentParamHead(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_components=max_components,
            num_types=num_types,
            num_continuous=num_continuous,
            head_mode=head_mode,
            center_representation=center_representation,
            center_x_bins=center_x_bins,
            center_y_bins=center_y_bins,
        )
        self.aux_center_head = aux_center_head
        if aux_center_head:
            if center_representation != "bin_offset":
                raise ValueError("aux_center_head requires center_representation='bin_offset'.")
            self.aux_center = AuxCenterHead(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                max_components=max_components,
                center_x_bins=center_x_bins,
                center_y_bins=center_y_bins,
            )
        else:
            self.aux_center = None

    def forward(self, signals: torch.Tensor, features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.feature_fusion_mode == "none":
            if self.encoder is None:
                raise ValueError("Signal encoder is absent.")
            latent = self.encoder(signals)
        elif self.feature_fusion_mode == "features_only":
            if features is None:
                raise ValueError("features are required when feature_fusion_mode='features_only'.")
            latent = self.feature_encoder(features)
        elif self.feature_fusion_mode == "concat_latent":
            if features is None:
                raise ValueError("features are required when feature_fusion_mode='concat_latent'.")
            if self.encoder is None:
                raise ValueError("Signal encoder is absent.")
            signal_latent = self.encoder(signals)
            feature_latent = self.feature_encoder(features)
            latent = self.fusion(torch.cat([signal_latent, feature_latent], dim=1))
        else:
            raise ValueError(f"Unsupported feature_fusion_mode: {self.feature_fusion_mode}")
        out = self.head(latent)
        if self.aux_center is not None:
            out.update(self.aux_center(latent))
        return out
