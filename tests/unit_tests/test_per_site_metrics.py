"""Unit tests for the per-site metric collector (#525, #441).

The collector closes the last hop of a transport that already works: clients
stream `fed.analytix_log_stats`, NVFlare relays it, and nothing on the server
handled it -- so `cross_val_results.json` was `{}` after every p2p run.

These tests use stub FLContext/DXO objects, so no NVFlare runtime, server or
network is needed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import SHARED_CUSTOM_DIR, import_module_from_path  # noqa: E402

pytest.importorskip("nvflare")

COLLECTOR_PATH = SHARED_CUSTOM_DIR / "per_site_metrics.py"


@pytest.fixture(scope="module")
def module():
    return import_module_from_path("_test_per_site_metrics", COLLECTOR_PATH)


class FakeCtx:
    """Minimal FLContext: records props and the events fired against it."""

    def __init__(self):
        self.props = {}
        self.events = []

    def set_prop(self, key, value, private=True, sticky=False):
        self.props[key] = value

    def get_prop(self, key, default=None):
        return self.props.get(key, default)


def _collector(module, **kwargs):
    c = module.PerSiteMetricCollector(**kwargs)
    # Widget logging expects a real engine; these tests only care about behaviour.
    c.log_info = lambda *a, **k: None
    c.log_warning = lambda *a, **k: None
    c.log_error = lambda *a, **k: None
    c.fire_event = lambda event_type, fl_ctx: fl_ctx.events.append(
        (event_type, dict(fl_ctx.props))
    )
    return c


def _metric(module, tag, value, step=0):
    """Build the shareable a client would stream for one metric."""
    from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType

    return AnalyticsData(key=tag, value=value, data_type=AnalyticsDataType.SCALAR,
                         global_step=step).to_dxo().to_shareable()


def test_publishes_one_record_per_site(module):
    c = _collector(module)
    ctx = FakeCtx()
    c.initialize(ctx)
    c.save(ctx, _metric(module, "val/AUC_ROC", 0.82), "UMCU_1")
    c.save(ctx, _metric(module, "val/AUC_ROC", 0.71), "CAM_1")
    c.finalize(ctx)

    assert len(ctx.events) == 2
    sites = {props["_data_client_"] for _, props in ctx.events}
    assert sites == {"UMCU_1", "CAM_1"}


def test_published_record_carries_support_alongside_the_metric(module):
    """The point of the change: a per-class metric never travels without its n."""
    from nvflare.apis.dxo import from_shareable

    c = _collector(module)
    ctx = FakeCtx()
    c.initialize(ctx)
    for tag, value in [
        ("val/AUC_ROC_class1", 0.372),
        ("val/support_class1", 2.0),
        ("val/n", 41.0),
    ]:
        c.save(ctx, _metric(module, tag, value), "UMCU_1")
    c.finalize(ctx)

    _, props = ctx.events[0]
    data = from_shareable(props["_validation_result_"]).data
    assert data["val/AUC_ROC_class1"] == pytest.approx(0.372)
    assert data["val/support_class1"] == pytest.approx(2.0)
    assert data["val/n"] == pytest.approx(41.0)


def test_training_noise_is_dropped(module):
    c = _collector(module)
    ctx = FakeCtx()
    c.initialize(ctx)
    c.save(ctx, _metric(module, "train/loss", 0.5), "UKA_1")
    c.finalize(ctx)
    assert ctx.events == []


def test_later_values_win(module):
    """Each site's final value is published, not its first."""
    from nvflare.apis.dxo import from_shareable

    c = _collector(module)
    ctx = FakeCtx()
    c.initialize(ctx)
    c.save(ctx, _metric(module, "val/AUC_ROC", 0.60, step=1), "UKA_1")
    c.save(ctx, _metric(module, "val/AUC_ROC", 0.82, step=9), "UKA_1")
    c.finalize(ctx)

    _, props = ctx.events[0]
    assert from_shareable(props["_validation_result_"]).data["val/AUC_ROC"] == pytest.approx(0.82)


def test_a_malformed_record_does_not_lose_the_others(module):
    """One bad site must not take down the run or the other sites' results."""
    c = _collector(module)
    ctx = FakeCtx()
    c.initialize(ctx)
    c.save(ctx, "not a shareable", "BROKEN_1")
    c.save(ctx, _metric(module, "val/AUC_ROC", 0.82), "UKA_1")
    c.finalize(ctx)

    assert len(ctx.events) == 1
    assert ctx.events[0][1]["_data_client_"] == "UKA_1"


def test_no_metrics_publishes_nothing(module):
    c = _collector(module)
    ctx = FakeCtx()
    c.initialize(ctx)
    c.finalize(ctx)
    assert ctx.events == []


def test_initialize_clears_state_between_runs(module):
    c = _collector(module)
    first = FakeCtx()
    c.initialize(first)
    c.save(first, _metric(module, "val/AUC_ROC", 0.82), "UKA_1")

    second = FakeCtx()
    c.initialize(second)          # START_RUN of the next job
    c.finalize(second)
    assert second.events == []


def test_model_owner_is_recorded(module):
    c = _collector(module, model_owner="swarm_global")
    ctx = FakeCtx()
    c.initialize(ctx)
    c.save(ctx, _metric(module, "val/AUC_ROC", 0.82), "UKA_1")
    c.finalize(ctx)
    assert ctx.events[0][1]["_model_owner_"] == "swarm_global"

def _metrics_batch(module, values, step=0):
    """The shape MLflowWriter.log_metrics sends: a dict under the tag "metrics".

    This is what ClientLogger produces, so it is what the Lightning training
    path actually puts on the wire.
    """
    from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType

    return AnalyticsData(key="metrics", value=values, data_type=AnalyticsDataType.METRICS,
                         global_step=step).to_dxo().to_shareable()


def test_batch_payload_from_the_lightning_path_is_recorded(module):
    """Regression: the dict shape was silently dropped, so nothing was published."""
    from nvflare.apis.dxo import from_shareable

    c = _collector(module)
    ctx = FakeCtx()
    c.initialize(ctx)
    c.save(ctx, _metrics_batch(module, {
        "val/AUC_ROC": 0.82,
        "val/AUC_ROC_class1": 0.372,
        "val/support_class1": 2.0,
        "val/n": 41.0,
        "train/loss": 0.5,          # filtered out
    }), "UMCU_1")
    c.finalize(ctx)

    assert len(ctx.events) == 1
    data = from_shareable(ctx.events[0][1]["_validation_result_"]).data
    assert data["val/AUC_ROC"] == pytest.approx(0.82)
    assert data["val/support_class1"] == pytest.approx(2.0)
    assert data["val/n"] == pytest.approx(41.0)
    assert "train/loss" not in data


def test_batch_and_scalar_payloads_merge(module):
    from nvflare.apis.dxo import from_shareable

    c = _collector(module)
    ctx = FakeCtx()
    c.initialize(ctx)
    c.save(ctx, _metrics_batch(module, {"val/AUC_ROC": 0.60}), "UKA_1")
    c.save(ctx, _metric(module, "val/ACC", 0.75), "UKA_1")
    c.finalize(ctx)

    data = from_shareable(ctx.events[0][1]["_validation_result_"]).data
    assert data["val/AUC_ROC"] == pytest.approx(0.60)
    assert data["val/ACC"] == pytest.approx(0.75)


def test_non_numeric_values_in_a_batch_are_skipped(module):
    from nvflare.apis.dxo import from_shareable

    c = _collector(module)
    ctx = FakeCtx()
    c.initialize(ctx)
    c.save(ctx, _metrics_batch(module, {"val/AUC_ROC": 0.8, "val/note": "n/a"}), "UKA_1")
    c.finalize(ctx)

    data = from_shareable(ctx.events[0][1]["_validation_result_"]).data
    assert "val/note" not in data
    assert data["val/AUC_ROC"] == pytest.approx(0.8)
