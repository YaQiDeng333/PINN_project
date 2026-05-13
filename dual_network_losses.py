"""Loss skeletons for the dual-network variational branch.

These helpers are intentionally minimal. The weak-form term still needs
compact-support test functions and explicit quadrature weights before it can
be treated as the branch's final material-update loss.
"""

import torch


def gradients(output, inputs, create_graph=True, retain_graph=True):
    """Compute first derivatives of output with respect to inputs."""
    if not inputs.requires_grad:
        raise ValueError(
            "coords must have requires_grad=True for PINN derivative losses"
        )

    return torch.autograd.grad(
        outputs=output,
        inputs=inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True,
    )[0]


def energy_loss(phi, mu, coords):
    """mean(0.5 * mu * (phi_x^2 + phi_y^2))."""
    grad_phi = gradients(phi, coords)
    phi_x = grad_phi[..., 0:1]
    phi_y = grad_phi[..., 1:2]
    return torch.mean(0.5 * mu * (phi_x.pow(2) + phi_y.pow(2)))


def data_loss(phi_probe, coords_probe, bz_meas):
    """mean((-phi_y - bz_meas)^2) on the probe line."""
    grad_phi = gradients(phi_probe, coords_probe)
    phi_y = grad_phi[..., 1:2]
    bz_pred = (-phi_y).reshape(-1, 1)
    bz_meas = bz_meas.reshape(-1, 1)

    if bz_pred.shape[0] != bz_meas.shape[0]:
        raise ValueError(
            "bz_pred and bz_meas must have the same number of probe points: "
            f"got {bz_pred.shape[0]} and {bz_meas.shape[0]}"
        )

    return torch.mean((bz_pred - bz_meas).pow(2))


def tv_loss(mu, coords, eps=1e-8):
    """Continuous-coordinate TV proxy: mean(sqrt(mu_x^2 + mu_y^2 + eps))."""
    grad_mu = gradients(mu, coords)
    return torch.mean(torch.sqrt(torch.sum(grad_mu.pow(2), dim=-1, keepdim=True) + eps))


def weak_form_loss(mu, phi_fixed, coords, test_grads=None, quad_weights=None):
    """Skeleton weak-form residual.

    This is not the final weak-form implementation. The branch still needs
    test functions v_q, compact support, quadrature weights, and normalization.
    """
    if test_grads is None:
        raise NotImplementedError(
            "weak_form_loss skeleton requires test_grads. "
            "Implement compact-support test function gradients first."
        )

    if test_grads.dim() == 2:
        if test_grads.shape[-1] != 2:
            raise ValueError("test_grads with shape [N, 2] must have last dim 2")
        test_grads = test_grads.unsqueeze(0)
    elif test_grads.dim() == 3:
        if test_grads.shape[-1] != 2:
            raise ValueError("test_grads with shape [Q, N, 2] must have last dim 2")
    else:
        raise ValueError("test_grads must have shape [N, 2] or [Q, N, 2]")

    if mu.shape[0] != coords.shape[0] or test_grads.shape[1] != coords.shape[0]:
        raise ValueError("mu, coords, and test_grads must use the same N points")

    # PhiNet parameters should be frozen in the mu-step training loop. Here
    # grad_phi is computed from phi_fixed with respect to coords, then detached
    # so the weak residual treats it as a fixed field coefficient.
    grad_phi = gradients(
        phi_fixed,
        coords,
        create_graph=False,
        retain_graph=False,
    ).detach()

    mu_values = mu.reshape(1, coords.shape[0], -1)
    if mu_values.shape[-1] != 1:
        raise ValueError("mu must be scalar per coordinate point")

    grad_phi = grad_phi.reshape(1, coords.shape[0], 2)
    integrand = mu_values * torch.sum(grad_phi * test_grads, dim=-1, keepdim=True)

    if quad_weights is None:
        residual_q = torch.mean(integrand, dim=1).squeeze(-1)
    else:
        if quad_weights.dim() == 1:
            quad_weights = quad_weights.reshape(1, -1, 1)
        elif quad_weights.dim() == 2:
            if quad_weights.shape[-1] != 1:
                raise ValueError("quad_weights with shape [N, 1] must have last dim 1")
            quad_weights = quad_weights.unsqueeze(0)
        elif quad_weights.dim() == 3:
            if quad_weights.shape[-1] != 1:
                raise ValueError("quad_weights with shape [Q, N, 1] must have last dim 1")
        else:
            raise ValueError("quad_weights must have shape [N], [N, 1], or [Q, N, 1]")

        if quad_weights.shape[1] != coords.shape[0]:
            raise ValueError("quad_weights must use the same N points as coords")
        if quad_weights.shape[0] not in (1, test_grads.shape[0]):
            raise ValueError("quad_weights first dimension must be 1 or Q")

        residual_q = torch.sum(integrand * quad_weights, dim=1).squeeze(-1)

    return torch.mean(residual_q.pow(2))
