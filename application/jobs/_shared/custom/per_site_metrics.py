"""Collect per-site training metrics on the coordinator (#525, #441).

Why this exists
---------------
Clients already stream their metrics server-ward: ``MetricRelay`` fires
``fed.analytix_log_stats``, ``ClientFedEventRunner`` relays it, and
``ServerFedEventRunner`` delivers it to server handlers. Nothing on the server
ever handled it, so ``cross_site_val/cross_val_results.json`` came back ``{}``
after every peer-to-peer run -- on the two-day 20-round run and on a 52-second
smoke run alike -- and per-site figures had to be gathered by logging into
individual hospitals by hand.

This receiver closes that last hop. It accumulates the streamed metrics per
site and, at end of run, hands them to the already-configured
``ValidationJsonGenerator`` by firing ``VALIDATION_RESULT_RECEIVED``. No client
change and no new transport are needed.

Reporting shape
---------------
Metrics are recorded with the support counts emitted alongside them by
``BaseModel.compute_epoch_metrics`` (``*/support_class{i}`` and ``*/n``). A
per-class metric must never travel without its support: ranking centres by a
bare macro AUROC, where one class had two positives, is what produced a wrong
conclusion about UMCU. See ``docs/EVALUATION_PITFALLS.md`` (E1).
"""

from threading import Lock

from nvflare.apis.analytix import AnalyticsData
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.widgets.streaming import AnalyticsReceiver

GLOBAL_MODEL_OWNER = "swarm_global"


class PerSiteMetricCollector(AnalyticsReceiver):
    """Accumulate streamed per-site metrics and publish them at end of run.

    Args:
        model_owner: label recorded as the owner of the evaluated model. In a
            swarm every site evaluates the shared model on its own data, so the
            default is a single global label rather than a peer's name.
        metric_prefixes: only metrics whose key starts with one of these are
            kept. Defaults to validation and test; training-time noise is
            dropped so the published summary stays readable.
    """

    def __init__(self, model_owner: str = GLOBAL_MODEL_OWNER, metric_prefixes=None):
        super().__init__()
        self._model_owner = model_owner
        self._prefixes = tuple(metric_prefixes or ("val/", "test/"))
        self._metrics = {}
        self._lock = Lock()

    def initialize(self, fl_ctx: FLContext):
        with self._lock:
            self._metrics = {}

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        """Record one streamed metric for ``record_origin``.

        Later values for the same key overwrite earlier ones, so what is
        published is each site's final value rather than its first.
        """
        try:
            data = AnalyticsData.from_dxo(from_shareable(shareable))
        except Exception as exc:
            # A malformed record from one site must not take down the run.
            self.log_warning(fl_ctx, f"Unparseable metric from {record_origin}: {exc}", fire_event=False)
            return
        if not data or not data.tag or not str(data.tag).startswith(self._prefixes):
            return

        value = data.value
        if hasattr(value, "item"):          # torch/np scalar
            try:
                value = value.item()
            except Exception:
                return
        if not isinstance(value, (int, float, bool)):
            return

        with self._lock:
            site = self._metrics.setdefault(record_origin, {})
            site[str(data.tag)] = value
            if data.step is not None:
                site["_last_step"] = data.step

    def finalize(self, fl_ctx: FLContext):
        """Publish one validation record per site.

        ``ValidationJsonGenerator`` catches these and writes
        ``cross_site_val/cross_val_results.json`` as
        ``{site: {model_owner: metrics}}``.
        """
        with self._lock:
            collected = {site: dict(values) for site, values in self._metrics.items()}

        if not collected:
            self.log_warning(
                fl_ctx,
                "No per-site metrics were received; cross_val_results.json will be empty. "
                "Check that the clients configure metric_relay with event_type "
                "'fed.analytix_log_stats'.",
                fire_event=False,
            )
            return

        for site, values in collected.items():
            dxo = DXO(data_kind=DataKind.METRICS, data=values)
            fl_ctx.set_prop(AppConstants.MODEL_OWNER, self._model_owner, private=True, sticky=False)
            fl_ctx.set_prop(AppConstants.DATA_CLIENT, site, private=True, sticky=False)
            fl_ctx.set_prop(
                AppConstants.VALIDATION_RESULT, dxo.to_shareable(), private=True, sticky=False
            )
            self.fire_event(AppEventType.VALIDATION_RESULT_RECEIVED, fl_ctx)

        self.log_info(
            fl_ctx,
            f"Published metrics for {len(collected)} site(s): {', '.join(sorted(collected))}",
            fire_event=False,
        )
