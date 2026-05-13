"""Minimal training entry skeleton for the dual-network variational branch."""

from dual_network_losses import data_loss, energy_loss, tv_loss, weak_form_loss
from dual_network_models import MuNet, PhiNet


DEFAULT_CONFIG = {
    "x_range": (-15.0, 15.0),
    "y_range": (0.0, 10.0),
    "y_s": 10.0,
    "mu_min": 1.0,
    "mu_max": 1000.0,
}


def main():
    print("Dual-network variational training skeleton.")
    print("This script does not start large-scale training.")
    print("It does not read data files or call train_pinn.py.")
    print("Future coords tensors must use coords.requires_grad_(True).")
    print("Future phi-step should freeze MuNet or pass mu.detach().")
    print("Future mu-step should freeze PhiNet and treat grad_phi as fixed.")
    print(f"Default config: {DEFAULT_CONFIG}")
    print(f"Models available: {PhiNet.__name__}, {MuNet.__name__}")
    print(
        "Loss helpers available: "
        f"{energy_loss.__name__}, {data_loss.__name__}, "
        f"{tv_loss.__name__}, {weak_form_loss.__name__}"
    )


if __name__ == "__main__":
    main()
