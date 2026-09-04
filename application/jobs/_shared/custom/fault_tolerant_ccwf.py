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
  rotating aggregator can tolerate a peer that submits a bad result when the
  configured minimum is lower than the trainer count. When every response is
  required, bad/rejected results and gather timeout instead fail the round and
  partial aggregation is prohibited.

Tolerance requires ``min_clients`` (server) and ``min_responses_required``
(client) to be lower than the number of participating clients. Setting both to
the full named client count selects strict all-site behavior instead.

In tolerant mode this changes wait-for-all semantics from #345 into "wait for
at least min_responses". In strict mode the controller retains every named
client and retries only unacknowledged learn-task deliveries; duplicate delivery
of an already accepted round is idempotent.
"""

import threading
import time
from datetime import datetime

from nvflare.apis.fl_constant import ReservedKey, ReservedTopic
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, make_reply
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.common import Constant, ResultType, status_report_from_dict
from nvflare.app_common.ccwf.server_ctl import ClientStatus
from nvflare.app_common.ccwf.swarm_client_ctl import Gatherer, SwarmClientController
from nvflare.app_common.ccwf.swarm_server_ctl import SwarmServerController
from nvflare.security.logging import secure_format_traceback

WARM_START_REQUIRED_MISSING = "WARM_START_REQUIRED_MISSING"


def is_non_tolerable_client_error(error) -> bool:
    return bool(error and WARM_START_REQUIRED_MISSING in str(error))


class _PermissionReplyRetryEngine:
    """Turn a missing submission-permission reply into a retryable response.

    ``SwarmClientController.do_learn_task`` already retries
    ``SERVICE_UNAVAILABLE`` replies, but treats a missing reply as a fatal client
    error. A control-message reply can arrive after the short request timeout on
    a relayed connection. Retrying is safe because ``Gatherer`` remembers a
    granted slot: another request from the same client returns ``OK`` without
    reserving a second slot.

    The adapter is installed on a cloned ``FLContext`` for one learning task, so
    it does not monkey-patch the shared engine. All non-permission engine calls
    and all explicit replies (including ``MODEL_UNRECOGNIZED``) are unchanged.
    """

    def __init__(self, engine, controller):
        self._engine = engine
        self._controller = controller
        self._missing_reply_count = 0

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def send_aux_request(self, *args, **kwargs):
        responses = self._engine.send_aux_request(*args, **kwargs)

        if kwargs.get("topic") != self._controller.request_to_submit_learn_result_task_name:
            return responses
        if not isinstance(responses, dict):
            # Preserve the stock failure path for malformed engine responses.
            return responses

        targets = kwargs.get("targets")
        if not isinstance(targets, (list, tuple)) or len(targets) != 1:
            return responses

        target = targets[0]
        # Shareable is dict-like and an explicit header-only reply can be
        # falsey. Presence, not truthiness, distinguishes it from a miss.
        if target in responses and responses[target] is not None:
            return responses

        self._missing_reply_count += 1
        # Log the first miss and then periodically; a long relay outage should
        # remain visible without producing a warning every retry interval.
        if self._missing_reply_count == 1 or self._missing_reply_count % 10 == 0:
            self._controller.log_warning(
                kwargs.get("fl_ctx"),
                f"missing submission-permission reply from {target}; treating it as transient and retrying "
                f"(miss {self._missing_reply_count})",
            )

        retryable = dict(responses)
        retryable[target] = make_reply(ReturnCode.SERVICE_UNAVAILABLE)
        return retryable


class _LearnScatterRetryEngine:
    """Retry a learn-task scatter without resending to acknowledged clients.

    ``TaskController.broadcast_and_wait`` converts every explicit non-OK reply
    into a generic ``ERROR``. This context-local adapter runs immediately below
    that controller, where the raw responses are still available. Only a
    missing response, ``None``, or an explicit ``TIMEOUT`` is retryable. Every
    other explicit response is retained as terminal, and a client that replied
    ``OK`` is never sent the same task again.
    """

    def __init__(self, engine, controller, deadline: float, attempt_timeout: float, retry_interval: float):
        self._engine = engine
        self._controller = controller
        self._deadline = deadline
        self._attempt_timeout = attempt_timeout
        self._retry_interval = retry_interval
        self._active = True

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def deactivate(self):
        # The cloned FLContext can be retained by the locally queued learn task.
        # Once scatter returns, make this adapter a transparent pass-through so
        # a later round cannot inherit the old round's deadline.
        self._active = False

    @staticmethod
    def _return_code(reply):
        if reply is None:
            return None
        getter = getattr(reply, "get_return_code", None)
        if callable(getter):
            return getter(ReturnCode.OK)
        if isinstance(reply, dict):
            # This fallback also keeps the adapter straightforward to unit test.
            return reply.get("return_code", reply.get(ReservedKey.RC, ReturnCode.OK))
        return ReturnCode.OK

    def _is_learn_scatter(self, kwargs) -> bool:
        if not self._active or kwargs.get("topic") != ReservedTopic.DO_TASK:
            return False
        request = kwargs.get("request")
        get_header = getattr(request, "get_header", None)
        return callable(get_header) and get_header(ReservedKey.TASK_NAME) == self._controller.do_learn_task_name

    def _is_aborted(self, fl_ctx) -> bool:
        if getattr(self._controller, "asked_to_stop", False):
            return True
        get_abort_signal = getattr(fl_ctx, "get_run_abort_signal", None)
        if not callable(get_abort_signal):
            return False
        abort_signal = get_abort_signal()
        return bool(abort_signal and getattr(abort_signal, "triggered", False))

    def _wait_before_retry(self, fl_ctx):
        wait_until = min(self._deadline, time.time() + self._retry_interval)
        while not self._is_aborted(fl_ctx):
            remaining = wait_until - time.time()
            if remaining <= 0:
                return
            time.sleep(min(0.2, remaining))

    def send_aux_request(self, *args, **kwargs):
        # TaskController uses keyword arguments for DO_TASK. Preserve unfamiliar
        # call shapes exactly rather than guessing argument positions.
        if not self._is_learn_scatter(kwargs):
            return self._engine.send_aux_request(*args, **kwargs)

        targets = kwargs.get("targets")
        if not isinstance(targets, (list, tuple)) or not targets:
            return self._engine.send_aux_request(*args, **kwargs)

        pending = list(targets)
        terminal_responses = {}
        last_timeout_responses = {}
        attempt = 0
        fl_ctx = kwargs.get("fl_ctx")
        request = kwargs.get("request")
        current_round = request.get_header(AppConstants.CURRENT_ROUND)

        while pending and not self._is_aborted(fl_ctx):
            remaining = self._deadline - time.time()
            if remaining <= 0:
                break

            attempt += 1
            attempt_kwargs = dict(kwargs)
            attempt_kwargs["targets"] = list(pending)
            attempt_kwargs["timeout"] = min(self._attempt_timeout, remaining)
            responses = self._engine.send_aux_request(*args, **attempt_kwargs)
            if not isinstance(responses, dict):
                # Let the stock controller handle a malformed engine response.
                return responses

            retry_targets = []
            for target in pending:
                if target not in responses or responses[target] is None:
                    retry_targets.append(target)
                    continue

                reply = responses[target]
                rc = self._return_code(reply)
                if rc == ReturnCode.TIMEOUT:
                    last_timeout_responses[target] = reply
                    retry_targets.append(target)
                else:
                    # OK and every explicit non-timeout error are terminal.
                    terminal_responses[target] = reply

            pending = retry_targets
            if not pending:
                break

            self._controller.log_warning(
                fl_ctx,
                f"learn-task scatter for round {current_round} attempt {attempt} did not acknowledge "
                f"{pending}; retrying only those clients",
            )
            if time.time() >= self._deadline:
                break
            self._wait_before_retry(fl_ctx)

        # Missing/None responses remain absent so the stock caller takes its
        # existing missing-reply failure path. Preserve a raw TIMEOUT when one
        # was received, and preserve every success/terminal error verbatim.
        final_responses = dict(terminal_responses)
        for target in pending:
            timeout_reply = last_timeout_responses.get(target)
            if timeout_reply is not None:
                final_responses[target] = timeout_reply
        return final_responses


class FaultTolerantGatherer(Gatherer):
    """Gatherer that tolerates a peer submitting a bad result instead of failing
    the whole gather. Faithful copy of ``Gatherer._do_gather`` with the return-code
    check moved above the response counting so a bad result is neither counted nor
    fatal."""

    def _all_responses_required(self) -> bool:
        return self.min_responses_required >= len(self.trainer_statuses)

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
            if self._all_responses_required():
                self.log_error(
                    fl_ctx,
                    f"Strict gather: bad result from required client {client_name} for round {result_round}: {rc}",
                )
                self.executor.update_status(action="gather", error=rc)
                return make_reply(rc)

            self.log_warning(
                fl_ctx,
                f"FaultTolerant: tolerating bad result from {client_name} for round {result_round}: "
                f"{rc} (not counted toward min_responses_required={self.min_responses_required})",
            )
            return make_reply(ReturnCode.OK)

        strict = self._all_responses_required()

        # Stock Gatherer counts a response before aggregator.accept() returns.
        # Keep that behavior in tolerant mode, where the minimum-response wait
        # is intentionally based on arrivals. In strict mode, marking the final
        # response here lets the monitor begin aggregation while the final
        # accept is still in progress. Count strict responses only after the
        # aggregator has accepted them below.
        if result_round == self.for_round and not strict:
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

        if strict and result_round == self.for_round and not accepted:
            # Preserve the stock BEFORE/AFTER event pair, but do not count a
            # rejected contribution as a valid strict response. Fail the run
            # instead of aggregating fewer than all required clients.
            self.executor.update_status(action="gather", error=ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if strict and result_round == self.for_round:
            # reply_time is the commit marker proving this contribution is
            # already present in the aggregator and all accept hooks completed.
            # is_done() may run concurrently, so this must be the final state
            # change before returning success.
            now = time.time()
            ts.reply_time = now
            if not self.min_resps_received_time:
                num_resps_received = sum(1 for status in self.trainer_statuses.values() if status.reply_time)
                if num_resps_received >= self.min_responses_required:
                    self.min_resps_received_time = now
        return make_reply(ReturnCode.OK)

    def is_done(self):
        if not self._all_responses_required():
            return super().is_done()

        missing = [name for name, status in self.trainer_statuses.items() if not status.reply_time]
        if not missing:
            return True

        if self.timeout and time.time() - self.start_time > self.timeout:
            # Never let _monitor_gather call aggregate() with a partial strict
            # round. Report the timeout once, then wait for the job abort path.
            if not getattr(self, "_strict_timeout_reported", False):
                self._strict_timeout_reported = True
                self.log_error(
                    self.fl_ctx,
                    f"strict gather for round {self.for_round} timed out waiting for required clients {missing}",
                )
                self.executor.update_status(action="gather_timeout", error=ReturnCode.TIMEOUT)
            return False

        return False


class FaultTolerantSwarmClientController(SwarmClientController):
    """SwarmClientController that uses FaultTolerantGatherer for aggregation."""

    def __init__(
        self,
        *args,
        broadcast_last_result: bool = True,
        learn_task_scatter_attempt_timeout=None,
        learn_task_scatter_retry_interval: float = 5.0,
        **kwargs,
    ):
        if learn_task_scatter_attempt_timeout is not None and learn_task_scatter_attempt_timeout <= 0:
            raise ValueError("learn_task_scatter_attempt_timeout must be positive")
        if learn_task_scatter_retry_interval <= 0:
            raise ValueError("learn_task_scatter_retry_interval must be positive")

        super().__init__(*args, **kwargs)
        self.broadcast_last_result = broadcast_last_result
        # Defaulting the per-attempt timeout to the stock ACK timeout preserves
        # existing jobs. A shorter explicit value permits retries while the
        # overall scatter remains bounded by learn_task_timeout.
        self.learn_task_scatter_attempt_timeout = (
            learn_task_scatter_attempt_timeout
            if learn_task_scatter_attempt_timeout is not None
            else self.learn_task_ack_timeout
        )
        self.learn_task_scatter_retry_interval = learn_task_scatter_retry_interval
        self._learn_request_dedupe_lock = threading.Lock()
        self._last_accepted_learn_round = None

    def start_run(self, fl_ctx: FLContext):
        # do_learn_task() instantiates the module-global ``Gatherer``; patch that
        # symbol so the aggregator role uses the fault-tolerant gatherer. (Done
        # here rather than copying the large do_learn_task method.) NVFlare runs
        # one workflow controller in each job child process; a future job that
        # deliberately mixes multiple swarm controller classes in one process
        # must replace this with an upstream gatherer factory/instance override.
        import nvflare.app_common.ccwf.swarm_client_ctl as _scc

        if _scc.Gatherer is not FaultTolerantGatherer:
            _scc.Gatherer = FaultTolerantGatherer
            self.log_info(fl_ctx, "installed strict/tolerant FaultTolerantGatherer policy")
        super().start_run(fl_ctx)

    def _scatter(self, task_data, for_round: int, fl_ctx: FLContext) -> bool:
        """Run the stock scatter with raw-response retry below TaskController."""
        engine = fl_ctx.get_engine()
        if not engine:
            return super()._scatter(task_data, for_round, fl_ctx)

        total_timeout = self.learn_task_timeout or self.learn_task_ack_timeout
        retry_engine = _LearnScatterRetryEngine(
            engine=engine,
            controller=self,
            deadline=time.time() + total_timeout,
            attempt_timeout=self.learn_task_scatter_attempt_timeout,
            retry_interval=self.learn_task_scatter_retry_interval,
        )
        retry_fl_ctx = fl_ctx.clone()
        retry_fl_ctx.put(
            key=ReservedKey.ENGINE,
            value=retry_engine,
            private=True,
            sticky=False,
        )
        try:
            return super()._scatter(task_data, for_round, retry_fl_ctx)
        finally:
            retry_engine.deactivate()

    def _try_process_learn_request(self, request, fl_ctx: FLContext):
        """Accept a round once and acknowledge a duplicate delivery safely."""
        current_round = request.get_header(AppConstants.CURRENT_ROUND)
        if current_round is None:
            return super()._try_process_learn_request(request, fl_ctx)

        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        sender = peer_ctx.get_identity_name()

        with self._learn_request_dedupe_lock:
            last_round = self._last_accepted_learn_round
            if last_round is not None:
                if current_round == last_round:
                    self.log_info(
                        fl_ctx,
                        f"duplicate Learn request from {sender} for round {current_round}; already accepted",
                    )
                    return make_reply(ReturnCode.OK)
                if current_round < last_round:
                    self.log_error(
                        fl_ctx,
                        f"stale Learn request from {sender} for round {current_round}; "
                        f"last accepted round is {last_round}",
                    )
                    return make_reply(ReturnCode.MODEL_UNRECOGNIZED)

            self.log_info(fl_ctx, f"Got Learn request from {sender}")
            if self.learn_task and not self.allow_busy_task:
                self.log_error(fl_ctx, f"got Learn request from {sender} while I'm still busy!")
                self.update_status(action="process_learn_request", error=ReturnCode.EXECUTION_EXCEPTION)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            self.log_info(fl_ctx, f"accepted learn request from {sender}")
            if not self.set_learn_task(task_data=request, fl_ctx=fl_ctx):
                self.log_error(fl_ctx, f"failed to queue Learn request from {sender} for round {current_round}")
                self.update_status(action="process_learn_request", error=ReturnCode.EXECUTION_EXCEPTION)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            self._last_accepted_learn_round = current_round
            return make_reply(ReturnCode.OK)

    def do_learn_task(self, name, task_data, fl_ctx: FLContext, abort_signal):
        """Run the stock learning task with retryable missing permission replies.

        The stock controller owns the permission loop. Supplying a context-local
        engine adapter keeps that implementation (including abort, max-wait, and
        explicit return-code handling) intact while changing only the missing-
        reply branch into its existing ``SERVICE_UNAVAILABLE`` retry branch.
        """
        engine = fl_ctx.get_engine()
        if not engine:
            return super().do_learn_task(name, task_data, fl_ctx, abort_signal)

        retry_fl_ctx = fl_ctx.clone()
        retry_fl_ctx.put(
            key=ReservedKey.ENGINE,
            value=_PermissionReplyRetryEngine(engine, self),
            private=True,
            sticky=False,
        )
        return super().do_learn_task(name, task_data, retry_fl_ctx, abort_signal)

    def _distribute_final_results(self, aggr_result, fl_ctx: FLContext):
        """Optionally skip duplicate LAST-result broadcast for validation jobs.

        Stock CCWF sends BEST and then LAST. In short two-client DL validation
        runs the BEST broadcast already verifies final model transfer; the
        duplicate LAST broadcast can keep the workflow open for the full
        final_result_ack_timeout when a peer is slow to service the second large
        transfer. Production jobs keep the stock behavior by default.
        """
        best_client = aggr_result.get_header(Constant.CLIENT)
        best_metric = aggr_result.get_header(Constant.METRIC)

        if best_client:
            if best_client == self.me:
                self.log_info(fl_ctx, f"I have global best metric {best_metric}")
                self.broadcast_final_result(
                    fl_ctx, ResultType.BEST, self.best_result, self.best_metric, self.best_round
                )
            else:
                try:
                    self._ask_to_share_best_result(best_client, best_metric, fl_ctx)
                except Exception:
                    self.log_error(
                        fl_ctx, f"error asking client {best_client} to share best result {secure_format_traceback()}"
                    )
        else:
            self.log_info(fl_ctx, "No global best result!")

        if self.broadcast_last_result:
            self.log_info(fl_ctx, "distributing last result")
            self.broadcast_final_result(fl_ctx, ResultType.LAST, self.last_result, round_num=self.last_round)
        else:
            self.log_info(fl_ctx, "skipping last result broadcast by configuration")
            self.update_status(action="finished_broadcast_last_result", all_done=True)


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
            if is_non_tolerable_client_error(report.error):
                self.asked_to_stop = True
                self.system_panic(
                    f"received non-tolerable warm-start failure report from client {client_name}: {report.error}",
                    fl_ctx,
                )
                return
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
