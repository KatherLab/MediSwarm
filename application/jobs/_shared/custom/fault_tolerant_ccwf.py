"""Fault-tolerant CCWF swarm controllers (issue #346).

Stock NVFlare CCWF aborts the entire wait-for-all run as soon as a *single*
client reports a transient failure -- a peer-comms ERROR, a stalled aggregator,
or a drop -> reconnect ``MODEL_UNRECOGNIZED`` desync. Over a multi-hour, many-site
run this turns one node's blip into total loss (observed repeatedly: UMCU r10,
VHIO r8/r1, USZ<-RSH r7, VHIO r10).

This module adds two contained, opt-in subclasses (referenced from the job
configs) that let a run survive one (or a few) such failures:

* ``FaultTolerantSwarmServerController`` -- on a client *error report* it prunes
  that client and continues, as long as ``min_clients`` would still be met,
  instead of ``system_panic``. (Silent/dropped clients and the configure /
  end-workflow counts are already tolerated by stock NVFlare when ``min_clients>0``.)
* ``FaultTolerantSwarmClientController`` + ``FaultTolerantGatherer`` -- the
  rotating aggregator tolerates a peer that submits a bad result: it is logged,
  not counted toward ``min_responses_required``, and acknowledged so the failure
  does not cascade. The gather completes once ``min_responses_required`` *good*
  results arrive (or ``learn_task_timeout`` fires).

REQUIRES (set in the job configs): ``min_clients`` (server) and
``min_responses_required`` (client) BOTH < the number of participating clients,
so there is headroom to drop a failing node. ``min_clients = 0`` (stock default)
means "all required" and disables tolerance.

NOTE: this changes wait-for-all semantics from #345 into "wait for at least
min_responses". It touches NVFlare-internal control flow and MUST be validated
with an induced single-node drop on a real multi-node run before production use.
A fully *re-synced* rejoin of a desynced node (vs. pruning it) is a follow-up;
here a desynced node is pruned and the run continues with the rest.
"""

import time
from datetime import datetime

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, make_reply
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.common import Constant, status_report_from_dict
from nvflare.app_common.ccwf.server_ctl import ClientStatus
from nvflare.app_common.ccwf.swarm_client_ctl import Gatherer, SwarmClientController
from nvflare.app_common.ccwf.swarm_server_ctl import SwarmServerController


class FaultTolerantGatherer(Gatherer):
    """Gatherer that tolerates a peer submitting a bad result instead of failing
    the whole gather. Faithful copy of ``Gatherer._do_gather`` with the return-code
    check moved above the response counting so a bad result is neither counted nor
    fatal."""

    def _do_gather(self, client_name: str, result, fl_ctx: FLContext):
        result_round = result.get_header(AppConstants.CURRENT_ROUND)
        ts = self.trainer_statuses.get(client_name)
        if not ts:
            self.log_error(
                fl_ctx, f"received result from {client_name} for round {result_round}, but it is not a trainer"
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if result_round > self.for_round:
            # this should never happen -- the peer is ahead of the gatherer
            self.log_error(
                fl_ctx,
                f"logic error: received result from {client_name} for round {result_round}, "
                f"which is > gatherer's current round {self.for_round}",
            )
            self.executor.update_status(action="gather", error=ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if result_round < self.for_round:
            self.log_warning(
                fl_ctx,
                f"received late result from {client_name} for round {result_round}, "
                f"which is < gatherer's current round {self.for_round}",
            )

        # FAULT TOLERANCE (#346): a peer whose local round failed submits a
        # bad-RC result. Do not abort the gather/run on one peer -- log it, do
        # NOT count it toward min_responses, and acknowledge (OK) so the failure
        # does not cascade into a fatal report. Moved above the counting block so
        # a bad result never sets reply_time.
        rc = result.get_return_code(ReturnCode.OK)
        if rc != ReturnCode.OK:
            self.log_warning(
                fl_ctx,
                f"FaultTolerant: tolerating bad result from {client_name} for round {result_round}: "
                f"{rc} (not counted toward min_responses_required={self.min_responses_required})",
            )
            return make_reply(ReturnCode.OK)

        if result_round == self.for_round:
            now = time.time()
            ts.reply_time = now
            if not self.min_resps_received_time:
                num_resps_received = 0
                for _, t in self.trainer_statuses.items():
                    if t.reply_time:
                        num_resps_received += 1
                if num_resps_received >= self.min_responses_required:
                    self.min_resps_received_time = now

        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self.for_round, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.TRAINING_RESULT, result, private=True, sticky=False)
        self.fire_event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT, fl_ctx)

        accepted = self.aggregator.accept(result, fl_ctx)
        accepted_msg = "ACCEPTED" if accepted else "REJECTED"
        self.log_info(
            fl_ctx, f"Contribution from {client_name} {accepted_msg} by the aggregator at round {result_round}."
        )

        fl_ctx.set_prop(AppConstants.AGGREGATION_ACCEPTED, accepted, private=True, sticky=False)
        self.fire_event(AppEventType.AFTER_CONTRIBUTION_ACCEPT, fl_ctx)
        return make_reply(ReturnCode.OK)


class FaultTolerantSwarmClientController(SwarmClientController):
    """SwarmClientController that uses FaultTolerantGatherer for aggregation."""

    def start_run(self, fl_ctx: FLContext):
        # do_learn_task() instantiates the module-global ``Gatherer``; patch that
        # symbol so the aggregator role uses the fault-tolerant gatherer. (Done
        # here rather than copying the large do_learn_task method.)
        import nvflare.app_common.ccwf.swarm_client_ctl as _scc

        if _scc.Gatherer is not FaultTolerantGatherer:
            _scc.Gatherer = FaultTolerantGatherer
            self.log_info(fl_ctx, "FaultTolerant: installed FaultTolerantGatherer (tolerates a failed peer in the gather)")
        super().start_run(fl_ctx)


class FaultTolerantSwarmServerController(SwarmServerController):
    """SwarmServerController that prunes a single failed client and continues,
    instead of panicking the whole run, as long as ``min_clients`` remains.

    Faithful copy of ``ServerSideController._update_client_status`` with only the
    ``report.error`` branch changed."""

    def _update_client_status(self, fl_ctx: FLContext):
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        client_name = peer_ctx.get_identity_name()

        reports = peer_ctx.get_prop(Constant.STATUS_REPORTS)
        if not reports:
            self.log_debug(fl_ctx, f"no status report from client {client_name}")
            return

        my_report = reports.get(self.workflow_id)
        if not my_report:
            return

        if client_name not in self.client_statuses:
            self.log_debug(
                fl_ctx, f"received status from client {client_name} not in active set (pruned or not yet configured)"
            )
            return

        report = status_report_from_dict(my_report)
        cs = self.client_statuses[client_name]
        assert isinstance(cs, ClientStatus)
        now = time.time()
        cs.last_report_time = now
        cs.num_reports += 1

        if report.error:
            remaining = len(self.client_statuses) - 1
            if self.min_clients and self.min_clients > 0 and remaining >= self.min_clients:
                # FAULT TOLERANCE (#346): tolerate one client's transient failure
                # (peer ERROR / MODEL_UNRECOGNIZED desync / drop) -- prune it and
                # continue; the rest still satisfy min_clients. A pruned client's
                # later reports are ignored (the "not in active set" branch above).
                self.log_warning(
                    fl_ctx,
                    f"FaultTolerant: client {client_name} reported error '{report.error}'; pruning and "
                    f"continuing with {remaining} active clients (min_clients={self.min_clients})",
                )
                del self.client_statuses[client_name]
                return
            self.asked_to_stop = True
            self.system_panic(
                f"received failure report from client {client_name}: {report.error} "
                f"(only {remaining} would remain, need min_clients={self.min_clients})",
                fl_ctx,
            )
            return

        if cs.status != report:
            cs.status = report
            cs.last_progress_time = now
            timestamp = datetime.fromtimestamp(report.timestamp) if report.timestamp else False
            self.log_info(
                fl_ctx,
                f"updated status of client {client_name} on round {report.last_round}: "
                f"timestamp={timestamp}, action={report.action}, all_done={report.all_done}",
            )
        else:
            self.log_debug(
                fl_ctx, f"ignored status report from client {client_name} at round {report.last_round}: no change"
            )
