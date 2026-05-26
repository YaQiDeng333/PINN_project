"""Smoke tests for signal-conditioned dual-network model skeletons."""

import torch

from conditional_dual_models import ConditionalDualNet


def _assert_shape(tensor, expected_shape, name):
    actual = tuple(tensor.shape)
    if actual != tuple(expected_shape):
        raise AssertionError(f"{name} shape mismatch: expected {expected_shape}, got {actual}")


def _assert_mu_bounds(mu):
    if torch.min(mu).item() < 1.0:
        raise AssertionError("mu minimum is below 1.0")
    if torch.max(mu).item() > 1000.0:
        raise AssertionError("mu maximum is above 1000.0")


def _assert_probability_bounds(prob, name):
    if torch.min(prob).item() < 0.0:
        raise AssertionError(f"{name} minimum is below 0.0")
    if torch.max(prob).item() > 1.0:
        raise AssertionError(f"{name} maximum is above 1.0")


def _expect_value_error(fn, label):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {label}")


def _check_forward_and_backward(
    model,
    signals,
    coords_2d,
    coords_3d,
    batch_size,
    num_coords,
    latent_dim,
    label,
    point_features=None,
):
    out_2d = model(signals, coords_2d, point_features=point_features)
    _assert_shape(out_2d["latent"], (batch_size, latent_dim), f"{label} latent with [N,2] coords")
    _assert_shape(out_2d["mu"], (batch_size, num_coords, 1), f"{label} mu with [N,2] coords")
    _assert_shape(out_2d["phi"], (batch_size, num_coords, 1), f"{label} phi with [N,2] coords")
    _assert_mu_bounds(out_2d["mu"])

    model.zero_grad(set_to_none=True)
    loss = out_2d["mu"].sum() + out_2d["phi"].sum()
    loss.backward()
    if not any(param.grad is not None for param in model.parameters()):
        raise AssertionError(f"{label} backward smoke test did not produce any parameter gradients")

    out_3d = model(signals, coords_3d, point_features=point_features)
    _assert_shape(out_3d["latent"], (batch_size, latent_dim), f"{label} latent with [B,N,2] coords")
    _assert_shape(out_3d["mu"], (batch_size, num_coords, 1), f"{label} mu with [B,N,2] coords")
    _assert_shape(out_3d["phi"], (batch_size, num_coords, 1), f"{label} phi with [B,N,2] coords")
    _assert_mu_bounds(out_3d["mu"])


def main():
    torch.manual_seed(48)

    batch_size = 3
    signal_len = 200
    num_coords = 500
    latent_dim = 32

    signals = torch.randn(batch_size, signal_len)
    coords_2d = torch.rand(num_coords, 2) * 2.0 - 1.0
    coords_3d = torch.rand(batch_size, num_coords, 2) * 2.0 - 1.0

    model = ConditionalDualNet(
        signal_len=signal_len,
        latent_dim=latent_dim,
        hidden_dim=64,
        num_layers=3,
    )
    _check_forward_and_backward(
        model,
        signals,
        coords_2d,
        coords_3d,
        batch_size,
        num_coords,
        latent_dim,
        "concat",
    )

    film_model = ConditionalDualNet(
        signal_len=signal_len,
        latent_dim=latent_dim,
        hidden_dim=64,
        num_layers=3,
        conditioning_mode="film",
    )
    _check_forward_and_backward(
        film_model,
        signals,
        coords_2d,
        coords_3d,
        batch_size,
        num_coords,
        latent_dim,
        "film",
    )

    cnn_concat_model = ConditionalDualNet(
        signal_len=signal_len,
        latent_dim=latent_dim,
        hidden_dim=64,
        num_layers=3,
        encoder_type="cnn",
        conditioning_mode="concat",
    )
    _check_forward_and_backward(
        cnn_concat_model,
        signals,
        coords_2d,
        coords_3d,
        batch_size,
        num_coords,
        latent_dim,
        "cnn+concat",
    )

    cnn_film_model = ConditionalDualNet(
        signal_len=signal_len,
        latent_dim=latent_dim,
        hidden_dim=64,
        num_layers=3,
        encoder_type="cnn",
        conditioning_mode="film",
    )
    _check_forward_and_backward(
        cnn_film_model,
        signals,
        coords_2d,
        coords_3d,
        batch_size,
        num_coords,
        latent_dim,
        "cnn+film",
    )

    point_features = torch.randn(batch_size, num_coords, 1)
    point_feature_model = ConditionalDualNet(
        signal_len=signal_len,
        latent_dim=latent_dim,
        hidden_dim=64,
        num_layers=3,
        extra_point_dim=1,
    )
    _check_forward_and_backward(
        point_feature_model,
        signals,
        coords_2d,
        coords_3d,
        batch_size,
        num_coords,
        latent_dim,
        "point_features",
        point_features=point_features,
    )

    mask_model = ConditionalDualNet(
        signal_len=signal_len,
        latent_dim=latent_dim,
        hidden_dim=64,
        num_layers=3,
        predict_mask=True,
    )
    mask_out = mask_model(signals, coords_2d)
    _assert_shape(mask_out["mask_logits"], (batch_size, num_coords, 1), "mask_logits")
    _assert_shape(mask_out["mask_prob"], (batch_size, num_coords, 1), "mask_prob")
    _assert_probability_bounds(mask_out["mask_prob"], "mask_prob")
    mask_model.zero_grad(set_to_none=True)
    mask_loss = mask_out["mask_logits"].sum() + mask_out["mu"].sum() + mask_out["phi"].sum()
    mask_loss.backward()
    if not any(param.grad is not None for param in mask_model.parameters()):
        raise AssertionError("mask head backward smoke test did not produce any parameter gradients")
    if "mask_logits" in model(signals, coords_2d):
        raise AssertionError("predict_mask=False should not return mask_logits")

    _expect_value_error(lambda: model(torch.randn(batch_size, signal_len, 1), coords_2d), "rank-3 signals")
    _expect_value_error(lambda: model(torch.randn(batch_size, signal_len + 1), coords_2d), "wrong signal_len")
    _expect_value_error(lambda: model(signals, torch.rand(num_coords, 3)), "rank-2 coords with wrong width")
    _expect_value_error(lambda: model(signals, torch.rand(batch_size + 1, num_coords, 2)), "coords batch mismatch")
    _expect_value_error(lambda: model(signals, torch.rand(batch_size, num_coords, 3)), "rank-3 coords with wrong width")
    _expect_value_error(lambda: model(signals, coords_2d, point_features=point_features), "unexpected point_features")
    _expect_value_error(lambda: point_feature_model(signals, coords_2d), "missing point_features")
    _expect_value_error(
        lambda: point_feature_model(signals, coords_2d, point_features=torch.randn(batch_size, num_coords, 2)),
        "wrong point_features feature dimension",
    )
    _expect_value_error(
        lambda: point_feature_model(signals, coords_2d, point_features=torch.randn(batch_size, num_coords + 1, 1)),
        "wrong point_features point dimension",
    )
    _expect_value_error(
        lambda: point_feature_model(signals, coords_2d, point_features=torch.randn(batch_size + 1, num_coords, 1)),
        "wrong point_features batch dimension",
    )
    _expect_value_error(
        lambda: ConditionalDualNet(signal_len=signal_len, conditioning_mode="bad_mode"),
        "invalid conditioning_mode",
    )
    _expect_value_error(
        lambda: ConditionalDualNet(signal_len=signal_len, encoder_type="bad_encoder"),
        "invalid encoder_type",
    )

    print("Conditional dual-network smoke test passed.")


if __name__ == "__main__":
    main()
