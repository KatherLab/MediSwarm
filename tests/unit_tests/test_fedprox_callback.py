"""
Unit tests for fedprox_callback.py — FedProx proximal term callback.

The FedProx callback adds a proximal term to gradient updates that penalises
the local model for deviating from the global model.  This regularises
local training in federated learning with non-IID data across sites.

The callback is compatible with both ODELIA (pytorch_lightning) and STAMP
(lightning) training pipelines via a try/except import pattern.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Ensure the shared custom directory is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_CUSTOM_DIR = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"


def _clear_fedprox_modules():
    """Remove cached fedprox_callback module so reimport works cleanly."""
    for mod_name in list(sys.modules):
        if mod_name in ("fedprox_callback",):
            del sys.modules[mod_name]


@pytest.fixture(autouse=True)
def _importable_fedprox():
    """Add shared custom dir to sys.path and clean up after."""
    original_path = sys.path[:]
    if str(SHARED_CUSTOM_DIR) not in sys.path:
        sys.path.insert(0, str(SHARED_CUSTOM_DIR))

    _clear_fedprox_modules()
    yield
    _clear_fedprox_modules()
    sys.path[:] = original_path


def _import_fedprox():
    _clear_fedprox_modules()
    import fedprox_callback
    return fedprox_callback


# ===================================================================
# Tests for FedProxCallback initialisation
# ===================================================================

class TestFedProxCallbackInit:
    """Test FedProxCallback constructor and validation."""

    def test_default_mu(self):
        mod = _import_fedprox()
        cb = mod.FedProxCallback()
        assert cb.mu == 0.01

    def test_custom_mu(self):
        mod = _import_fedprox()
        cb = mod.FedProxCallback(mu=0.5)
        assert cb.mu == 0.5

    def test_zero_mu(self):
        mod = _import_fedprox()
        cb = mod.FedProxCallback(mu=0.0)
        assert cb.mu == 0.0

    def test_negative_mu_raises(self):
        mod = _import_fedprox()
        with pytest.raises(ValueError, match="mu must be >= 0"):
            mod.FedProxCallback(mu=-0.1)

    def test_global_params_empty_at_init(self):
        mod = _import_fedprox()
        cb = mod.FedProxCallback()
        assert len(cb._global_params) == 0


# ===================================================================
# Tests for on_train_start — global model snapshot
# ===================================================================

class TestOnTrainStart:
    """Test on_train_start snapshots the global model."""

    def _make_module(self):
        """Create a mock Lightning module with real parameters."""
        module = MagicMock()
        p1 = torch.nn.Parameter(torch.randn(4, 4))
        p2 = torch.nn.Parameter(torch.randn(3))
        # A frozen parameter (requires_grad=False) should be excluded
        p3 = torch.nn.Parameter(torch.randn(2), requires_grad=False)
        module.named_parameters.return_value = [
            ("layer1.weight", p1),
            ("layer1.bias", p2),
            ("frozen_param", p3),
        ]
        return module, {"layer1.weight": p1, "layer1.bias": p2}

    def test_captures_trainable_params(self):
        mod = _import_fedprox()
        cb = mod.FedProxCallback(mu=0.1)
        module, _ = self._make_module()

        cb.on_train_start(MagicMock(), module)

        assert "layer1.weight" in cb._global_params
        assert "layer1.bias" in cb._global_params

    def test_excludes_frozen_params(self):
        mod = _import_fedprox()
        cb = mod.FedProxCallback(mu=0.1)
        module, _ = self._make_module()

        cb.on_train_start(MagicMock(), module)

        assert "frozen_param" not in cb._global_params

    def test_snapshots_are_detached_copies(self):
        """Snapshots should be independent of the original parameters."""
        mod = _import_fedprox()
        cb = mod.FedProxCallback(mu=0.1)
        module, params = self._make_module()

        cb.on_train_start(MagicMock(), module)

        # Modify original param — snapshot should not change
        original_snapshot = cb._global_params["layer1.weight"].clone()
        params["layer1.weight"].data.fill_(999.0)
        assert torch.equal(cb._global_params["layer1.weight"], original_snapshot)

    def test_snapshot_count_matches_trainable(self):
        mod = _import_fedprox()
        cb = mod.FedProxCallback(mu=0.1)
        module, _ = self._make_module()

        cb.on_train_start(MagicMock(), module)

        # 2 trainable params (layer1.weight and layer1.bias), not 3
        assert len(cb._global_params) == 2


# ===================================================================
# Tests for on_after_backward — proximal gradient injection
# ===================================================================

class TestOnAfterBackward:
    """Test on_after_backward adds proximal term to gradients."""

    def _setup(self, mu=0.1):
        """Create callback with a mock module and captured global params."""
        mod = _import_fedprox()
        cb = mod.FedProxCallback(mu=mu)

        # Create real parameters with gradients
        w = torch.nn.Parameter(torch.randn(4, 4))
        b = torch.nn.Parameter(torch.randn(3))

        module = MagicMock()
        module.named_parameters.return_value = [
            ("layer1.weight", w),
            ("layer1.bias", b),
        ]

        # Snapshot the global model (simulates start of round)
        cb.on_train_start(MagicMock(), module)

        # Now modify params to simulate local training
        w.data.add_(torch.randn_like(w.data) * 0.5)
        b.data.add_(torch.randn_like(b.data) * 0.5)

        # Simulate backward pass — set some gradients
        w.grad = torch.zeros_like(w)
        b.grad = torch.zeros_like(b)

        return cb, module, w, b

    def test_adds_proximal_gradient(self):
        cb, module, w, b = self._setup(mu=0.1)
        global_w = cb._global_params["layer1.weight"]

        cb.on_after_backward(MagicMock(), module)

        # Gradient should be mu * (w_local - w_global)
        expected = 0.1 * (w.data - global_w)
        assert torch.allclose(w.grad, expected, atol=1e-6)

    def test_bias_gets_proximal_gradient(self):
        cb, module, w, b = self._setup(mu=0.1)
        global_b = cb._global_params["layer1.bias"]

        cb.on_after_backward(MagicMock(), module)

        expected = 0.1 * (b.data - global_b)
        assert torch.allclose(b.grad, expected, atol=1e-6)

    def test_zero_mu_no_gradient_change(self):
        cb, module, w, b = self._setup(mu=0.0)
        original_grad = w.grad.clone()

        cb.on_after_backward(MagicMock(), module)

        # With mu=0, gradients should remain zero
        assert torch.equal(w.grad, original_grad)

    def test_accumulates_with_existing_gradients(self):
        """Proximal term should add to existing gradients, not replace them."""
        cb, module, w, b = self._setup(mu=0.1)
        global_w = cb._global_params["layer1.weight"]

        # Set some existing gradient (from main backward pass)
        existing_grad = torch.randn_like(w)
        w.grad = existing_grad.clone()

        cb.on_after_backward(MagicMock(), module)

        expected = existing_grad + 0.1 * (w.data - global_w)
        assert torch.allclose(w.grad, expected, atol=1e-6)

    def test_no_grad_param_skipped(self):
        """Parameters without .grad should be skipped (no error)."""
        mod = _import_fedprox()
        cb = mod.FedProxCallback(mu=0.1)

        w = torch.nn.Parameter(torch.randn(4))
        module = MagicMock()
        module.named_parameters.return_value = [("w", w)]

        cb.on_train_start(MagicMock(), module)
        w.data.add_(0.5)
        # w.grad is None — should not crash
        cb.on_after_backward(MagicMock(), module)

        assert w.grad is None

    def test_empty_global_params_is_noop(self):
        """If on_train_start was never called, on_after_backward is a no-op."""
        mod = _import_fedprox()
        cb = mod.FedProxCallback(mu=0.1)

        w = torch.nn.Parameter(torch.randn(4))
        w.grad = torch.zeros_like(w)
        module = MagicMock()
        module.named_parameters.return_value = [("w", w)]

        original_grad = w.grad.clone()
        cb.on_after_backward(MagicMock(), module)

        assert torch.equal(w.grad, original_grad)

    def test_large_mu_strong_regularisation(self):
        """With large mu, proximal gradient dominates."""
        cb, module, w, b = self._setup(mu=10.0)
        global_w = cb._global_params["layer1.weight"]

        cb.on_after_backward(MagicMock(), module)

        expected = 10.0 * (w.data - global_w)
        assert torch.allclose(w.grad, expected, atol=1e-6)

    def test_proximal_term_is_zero_when_params_equal(self):
        """When local params haven't changed, proximal gradient is zero."""
        mod = _import_fedprox()
        cb = mod.FedProxCallback(mu=0.5)

        w = torch.nn.Parameter(torch.randn(4))
        module = MagicMock()
        module.named_parameters.return_value = [("w", w)]

        # Snapshot — params are same as current
        cb.on_train_start(MagicMock(), module)

        # Don't modify params — set gradients to zero
        w.grad = torch.zeros_like(w)

        cb.on_after_backward(MagicMock(), module)

        # Proximal gradient should be zero since w == w_global
        assert torch.allclose(w.grad, torch.zeros_like(w), atol=1e-7)
