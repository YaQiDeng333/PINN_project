"""Smoke test for COMSOL parametric inverse model modules."""

from __future__ import annotations

import torch

from comsol_parametric_inverse_models import ParametricInverseNet


def _check_model(
    encoder_type: str,
    head_mode: str,
    signal_len: int = 600,
    feature_dim: int = 0,
    feature_fusion_mode: str = "none",
    center_representation: str = "continuous",
    center_x_bins: int = 0,
    center_y_bins: int = 0,
    aux_center_head: bool = False,
) -> None:
    model = ParametricInverseNet(
        signal_len=signal_len,
        hidden_dim=32,
        latent_dim=16,
        max_components=3,
        num_types=4,
        num_continuous=6,
        num_layers=2,
        encoder_type=encoder_type,
        head_mode=head_mode,
        feature_dim=feature_dim,
        feature_fusion_mode=feature_fusion_mode,
        center_representation=center_representation,
        center_x_bins=center_x_bins,
        center_y_bins=center_y_bins,
        aux_center_head=aux_center_head,
    )
    signals = torch.randn(4, signal_len)
    features = torch.randn(4, feature_dim) if feature_dim else None
    out = model(signals, features)
    assert out["presence_logits"].shape == (4, 3)
    assert out["presence_prob"].shape == (4, 3)
    assert out["type_logits"].shape == (4, 3, 4)
    assert out["continuous"].shape == (4, 3, 6)
    if center_representation == "continuous":
        assert "center_x_bin_logits" not in out
        assert "center_y_bin_logits" not in out
        assert "center_offset" not in out
        assert "aux_center_x_bin_logits" not in out
    else:
        assert out["center_x_bin_logits"].shape == (4, 3, center_x_bins)
        assert out["center_y_bin_logits"].shape == (4, 3, center_y_bins)
        assert out["center_offset"].shape == (4, 3, 2)
        if aux_center_head:
            assert out["aux_center_x_bin_logits"].shape == (4, 3, center_x_bins)
            assert out["aux_center_y_bin_logits"].shape == (4, 3, center_y_bins)
            assert out["aux_center_offset"].shape == (4, 3, 2)
        else:
            assert "aux_center_x_bin_logits" not in out
            assert "aux_center_y_bin_logits" not in out
            assert "aux_center_offset" not in out
    loss = (
        out["presence_logits"].mean()
        + out["type_logits"].mean()
        + out["continuous"].pow(2).mean()
    )
    if center_representation == "bin_offset":
        loss = loss + out["center_x_bin_logits"].mean() + out["center_y_bin_logits"].mean() + out["center_offset"].pow(2).mean()
        if aux_center_head:
            loss = (
                loss
                + out["aux_center_x_bin_logits"].mean()
                + out["aux_center_y_bin_logits"].mean()
                + out["aux_center_offset"].pow(2).mean()
            )
    loss.backward()


def main() -> None:
    torch.manual_seed(0)
    _check_model("mlp", "shared", signal_len=600)
    _check_model("cnn1d", "shared", signal_len=60)
    _check_model("cnn1d", "component_specific", signal_len=60)
    _check_model("cnn1d_attention", "component_specific", signal_len=20)
    _check_model("mlp", "shared", signal_len=60, feature_dim=8, feature_fusion_mode="features_only")
    _check_model("mlp", "shared", signal_len=60, feature_dim=8, feature_fusion_mode="concat_latent")
    _check_model("mlp", "shared", signal_len=60, center_representation="bin_offset", center_x_bins=5, center_y_bins=4)
    _check_model("cnn1d", "component_specific", signal_len=60, center_representation="bin_offset", center_x_bins=5, center_y_bins=4)
    _check_model("mlp", "shared", signal_len=60, center_representation="bin_offset", center_x_bins=5, center_y_bins=4, aux_center_head=True)
    try:
        ParametricInverseNet(signal_len=60, center_representation="bin_offset")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for missing center bins.")
    try:
        ParametricInverseNet(signal_len=60, aux_center_head=True)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for aux center head without bin_offset centers.")
    model = ParametricInverseNet(signal_len=600)
    try:
        model(torch.randn(4, 3, 200))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-flattened signals.")
    feature_model = ParametricInverseNet(signal_len=60, feature_dim=8, feature_fusion_mode="features_only")
    try:
        feature_model(torch.randn(4, 60))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for missing features.")
    try:
        feature_model(torch.randn(4, 60), torch.randn(4, 7))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for wrong feature_dim.")
    print("COMSOL parametric inverse model smoke test passed.")


if __name__ == "__main__":
    main()
