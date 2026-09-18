"""FedProx proximal term callback for PyTorch Lightning.

Implements FedProx (Li et al., MLSys 2020) as a Lightning Callback,
compatible with both ODELIA and STAMP training pipelines.

FedProx adds a proximal term to the loss that penalises the local model
for deviating from the global model received at the start of each round:

    L_fedprox = L_original + (mu / 2) * ||w_local - w_global||^2

This regularises local training to reduce client drift in federated
learning with heterogeneous (non-IID) data across sites.

Usage:
    Set environment variable FEDPROX_MU (ODELIA) or STAMP_FEDPROX_MU (STAMP)
    to a positive float to enable.  A value of 0 disables FedProx.

    Typical range: 0.001 (light regularisation) to 1.0 (strong).
    Recommended starting point: 0.01.
"""

import logging
from typing import Any

import torch

try:
    from lightning.pytorch.callbacks import Callback
except ImportError:
    from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class FedProxCallback(Callback):
    """Lightning Callback that adds a FedProx proximal loss term.

    At the start of each ``trainer.fit()`` call (i.e., each federated round),
    a frozen copy of the current model parameters is saved.  After each
    backward pass, the gradient of the proximal penalty
    ``(mu/2) * ||w - w_global||^2`` is added to parameter gradients.

    The proximal gradient for each parameter is simply:
        ``mu * (w_local - w_global)``

    This is the Lightning-native equivalent of NVFlare's ``PTFedProxLoss``,
    adapted so it works with models whose ``training_step`` returns the loss
    (rather than requiring manual loss composition in a raw training loop).

    Adding gradients in ``on_after_backward`` avoids in-place modification
    of the loss tensor, which can cause autograd issues.
    """

    def __init__(self, mu: float = 0.01):
        """
        Args:
            mu: FedProx proximal term weight.  Must be >= 0.
                0.0 effectively disables the proximal term.
        """
        super().__init__()
        if mu < 0:
            raise ValueError(f"FedProx mu must be >= 0, got {mu}")
        self.mu = mu
        self._global_params: dict[str, torch.Tensor] = {}

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        """Snapshot the global model at the beginning of each fit() call."""
        self._global_params = {
            name: param.detach().clone()
            for name, param in pl_module.named_parameters()
            if param.requires_grad
        }
        logger.info(
            f"FedProx: captured global model snapshot "
            f"({len(self._global_params)} param groups, mu={self.mu})"
        )

    def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
        """Add proximal term gradients after the main backward pass.

        The gradient of ``(mu/2) * ||w - w_global||^2`` w.r.t. ``w`` is
        ``mu * (w - w_global)``.  We add this directly to each parameter's
        ``.grad`` after the main loss backward has been computed.  This is
        mathematically equivalent to computing
        ``loss = L_original + (mu/2) * ||w - w_global||^2`` and calling
        ``.backward()`` once, but avoids modifying the loss tensor.
        """
        if self.mu <= 0 or not self._global_params:
            return

        for name, param in pl_module.named_parameters():
            if name in self._global_params and param.grad is not None:
                ref = self._global_params[name].to(param.device)
                param.grad.add_(self.mu * (param.data - ref))
