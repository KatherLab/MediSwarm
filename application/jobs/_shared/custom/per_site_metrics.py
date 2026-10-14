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

Which model a number belongs to (release 1.9.0)
-----------------------------------------------
Up to 1.8 every ``val/*`` value was filed under one owner, ``swarm_global``,
and the last value per key won. The received global and the local model logged
the same keys, so the published numbers were each site's LOCAL model after its
last epoch, labelled as the global (October run 7a678d9a). The training code
now streams each model under its own prefix, and this collector files each
prefix under its own owner in ``cross_val_results.json``:

    site_local_model       val/*                       the site's own model, validated inside the fit
                                                       (last epoch of the last training round)
    swarm_global_received  global_val/*                the global each site received, validated at the
                                                       start of a round (the last round seen)
    swarm_global           final_val/*, final_test/*   the job's final global, scored by every site in the
                                                       evaluation-only round (ODELIA_FINAL_EVAL_ROUND=1)

Each owner carries ``<prefix>/round`` (the round its last values come from) and
``_last_step``. ``swarm_global`` is absent unless the job ran the evaluation-only
round: without it the final global is never evaluated at the sites.
"""

from threading import Lock

from nvflare.apis.analytix import AnalyticsData
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.widgets.streaming import AnalyticsReceiver

SITE_LOCAL_OWNER = "site_local_model"
GLOBAL_RECEIVED_OWNER = "swarm_global_received"
FINAL_GLOBAL_OWNER = "swarm_global"

# Metric-key prefix -> owner. Keys with any other prefix (train/*, epoch, ...) are dropped so the
# published summary stays readable.
DEFAULT_PREFIX_OWNERS = {
    "val/": SITE_LOCAL_OWNER,
    "global_val/": GLOBAL_RECEIVED_OWNER,
    "final_val/": FINAL_GLOBAL_OWNER,
    "final_test/": FINAL_GLOBAL_OWNER,
}


class PerSiteMetricCollector(AnalyticsReceiver):
    """Accumulate streamed per-site metrics and publish them at end of run.

    Args:
        prefix_owners: metric-key prefix -> owner label it is published under. Defaults to
            DEFAULT_PREFIX_OWNERS (see the module docstring). A key matching no prefix is dropped.
    """

    def __init__(self, prefix_owners=None):
        super().__init__()
        owners = dict(prefix_owners or DEFAULT_PREFIX_OWNERS)
        # Longest prefix first, so a prefix that extends another one wins.
        self._prefix_owners = tuple(sorted(owners.items(), key=lambda item: -len(item[0])))
        self._metrics = {}
        self._published = False
        self._lock = Lock()

    def _owner_of(self, key: str):
        for prefix, owner in self._prefix_owners:
            if key.startswith(prefix):
                return owner
        return None

    def initialize(self, fl_ctx: FLContext):
        with self._lock:
            self._metrics = {}
            self._published = False

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Publish on ABOUT_TO_END_RUN rather than waiting for END_RUN.

        ``ValidationJsonGenerator`` dumps ``cross_val_results.json`` inside its own
        ``END_RUN`` handler, and within one event the components run in the order
        they are listed in ``config_fed_server.conf``. Publishing on ``END_RUN``
        therefore only works when this collector happens to be listed first: with
        the generator first, the file is written empty and the metrics land in its
        dict a moment too late. That is exactly what job 7c6e72c6 did -- all eight
        sites reported, and the JSON was still ``{}``.

        ``ABOUT_TO_END_RUN`` is fired strictly before ``END_RUN``, so publishing
        there removes the dependency on component order. ``END_RUN`` is still
        handled as a fallback for any path that skips the earlier event;
        ``finalize`` is idempotent, so the second pass is a no-op.
        """
        if event_type == EventType.ABOUT_TO_END_RUN:
            self._handle_end_run_event(fl_ctx)
            return
        super().handle_event(event_type, fl_ctx)

    @staticmethod
    def _as_scalar(value):
        """Coerce a streamed value to a plain number, or None if it is not one."""
        if hasattr(value, "item"):          # torch / numpy scalar
            try:
                value = value.item()
            except Exception:
                return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        """Record streamed metrics for ``record_origin``.

        Two payload shapes arrive, depending on which writer the training side
        uses. ``MLflowWriter.log_metrics`` -- what ``ClientLogger`` calls, and so
        what the Lightning path produces -- sends a whole dict under the single
        tag ``"metrics"``; ``log_metric`` and ``SummaryWriter`` send one scalar
        per record. Both are handled, because dropping the batch shape would
        silently discard everything the Lightning trainer reports.

        Later values for the same key overwrite earlier ones, so what is
        published is each site's final value rather than its first. Values are
        kept per owner (see the module docstring), so the local model's values
        can never overwrite or stand in for a global's.
        """
        try:
            data = AnalyticsData.from_dxo(from_shareable(shareable))
        except Exception as exc:
            # A malformed record from one site must not take down the run.
            self.log_warning(fl_ctx, f"Unparseable metric from {record_origin}: {exc}", fire_event=False)
            return
        if not data:
            return

        if isinstance(data.value, dict):
            pairs = [(str(k), self._as_scalar(v)) for k, v in data.value.items()]
        else:
            pairs = [(str(data.tag or ""), self._as_scalar(data.value))]

        incoming = {}
        for key, value in pairs:
            owner = self._owner_of(key)
            if owner is not None and value is not None:
                incoming.setdefault(owner, {})[key] = value
        if not incoming:
            return

        with self._lock:
            site = self._metrics.setdefault(record_origin, {})
            for owner, values in incoming.items():
                record = site.setdefault(owner, {})
                record.update(values)
                if data.step is not None:
                    record["_last_step"] = data.step

    def finalize(self, fl_ctx: FLContext):
        """Publish one validation record per (site, owner).

        ``ValidationJsonGenerator`` catches these and writes
        ``cross_site_val/cross_val_results.json`` as
        ``{site: {owner: metrics}}``, owners as in the module docstring.
        """
        if self._published:
            return
        self._published = True

        with self._lock:
            collected = {
                site: {owner: dict(values) for owner, values in owners.items()}
                for site, owners in self._metrics.items()
            }

        if not collected:
            self.log_warning(
                fl_ctx,
                "No per-site metrics were received; cross_val_results.json will be empty. "
                "Check that the clients configure metric_relay with event_type "
                "'fed.analytix_log_stats'.",
                fire_event=False,
            )
            return

        for site, owners in sorted(collected.items()):
            for owner, values in sorted(owners.items()):
                dxo = DXO(data_kind=DataKind.METRICS, data=values)
                fl_ctx.set_prop(AppConstants.MODEL_OWNER, owner, private=True, sticky=False)
                fl_ctx.set_prop(AppConstants.DATA_CLIENT, site, private=True, sticky=False)
                fl_ctx.set_prop(
                    AppConstants.VALIDATION_RESULT, dxo.to_shareable(), private=True, sticky=False
                )
                self.fire_event(AppEventType.VALIDATION_RESULT_RECEIVED, fl_ctx)

        self.log_info(
            fl_ctx,
            f"Published metrics for {len(collected)} site(s): "
            + "; ".join(f"{site} ({', '.join(sorted(owners))})" for site, owners in sorted(collected.items())),
            fire_event=False,
        )
