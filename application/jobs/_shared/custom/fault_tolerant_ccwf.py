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

Pruning is *announced* (#595). The server is the only party that knows a client
is gone -- an error report, a ``CLIENT_DISCONNECTED`` event, or a client that
never answered the configure task -- but until now it kept that to itself. The
clients' trainer / aggregator / result lists are fixed at configure time, so
every later round still scattered to the dead site (retried for up to a day)
and every aggregator still waited ``wait_time_after_min_resps_received`` (an
hour) for its result. The first fault-injection test (four clients, one stopped
in round 2) therefore ran six minutes of training and sixty minutes of waiting
per round, and a worker that crashed at launch hung the start task for an hour.
Now the server sends a ``swarm_ft.prune.<workflow>`` aux notice to the survivors
whenever it prunes, and they drop the client from their lists and from a
gatherer that is still waiting on it. A pruned client does not rejoin.

If the pruned client was the *aggregator of the current round*, the round would
otherwise be lost: every trainer keeps asking it for permission to submit (the
stock loop retries a timed-out permission request forever). The survivors
therefore elect a replacement deterministically (the first remaining aggregator
candidate, the same on every client), the replacement sets up the round's
gatherer from the learn task it already holds, pending permission requests and
result sends are redirected to it, and a trainer whose result had already been
accepted by the dead aggregator submits it again.
"""

import threading
import time
from datetime import datetime

from nvflare.apis.controller_spec import Task
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReservedTopic
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.common import Constant, ResultType, status_report_from_dict
from nvflare.app_common.ccwf.server_ctl import ClientStatus
from nvflare.app_common.ccwf.swarm_client_ctl import Gatherer, SwarmClientController
from nvflare.app_common.ccwf.swarm_server_ctl import SwarmServerController
from nvflare.security.logging import secure_format_traceback

WARM_START_REQUIRED_MISSING = "WARM_START_REQUIRED_MISSING"

# Server -> surviving clients: "these clients are out of the swarm" (#595).
PRUNE_TOPIC_PREFIX = "swarm_ft.prune"
PRUNE_KEY_PRUNED = "pruned_clients"
PRUNE_KEY_ACTIVE = "active_clients"
PRUNE_KEY_REASON = "reason"


def prune_topic(workflow_id) -> str:
    return f"{PRUNE_TOPIC_PREFIX}.{workflow_id}"


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
        is_permission = kwargs.get("topic") == self._controller.request_to_submit_learn_result_task_name
        targets = kwargs.get("targets")
        single = isinstance(targets, (list, tuple)) and len(targets) == 1
        redirected_from = None
        if is_permission and single:
            replacement = self._controller.aggregator_replacement(targets[0])
            if replacement and replacement != targets[0]:
                # The round's aggregator was pruned (#595): ask its replacement instead,
                # and answer under the name the stock loop is waiting for.
                redirected_from = targets[0]
                if replacement == self._controller.me:
                    reply = self._controller.local_permission(kwargs.get("request"), kwargs.get("fl_ctx"))
                    responses = {redirected_from: reply}
                else:
                    kwargs = dict(kwargs)
                    kwargs["targets"] = [replacement]
                    responses = self._engine.send_aux_request(*args, **kwargs)
                    if isinstance(responses, dict):
                        responses = dict(responses)
                        responses[redirected_from] = responses.pop(replacement, None)
                if isinstance(responses, dict) and self._return_code(responses.get(redirected_from)) == ReturnCode.OK:
                    self._controller.note_permission_granted_by(replacement)
            else:
                responses = self._engine.send_aux_request(*args, **kwargs)
        else:
            responses = self._engine.send_aux_request(*args, **kwargs)

        if not is_permission:
            return responses
        if not isinstance(responses, dict):
            # Preserve the stock failure path for malformed engine responses.
            return responses
        if not single:
            return responses

        target = targets[0]
        if redirected_from is None and self._return_code(responses.get(target)) == ReturnCode.OK:
            self._controller.note_permission_granted_by(target)
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

    @staticmethod
    def _return_code(reply):
        if reply is None:
            return None
        getter = getattr(reply, "get_return_code", None)
        if callable(getter):
            return getter(ReturnCode.OK)
        if isinstance(reply, dict):
            return reply.get("return_code", reply.get(ReservedKey.RC, ReturnCode.OK))
        return ReturnCode.OK


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

    def drop_trainers(self, names, fl_ctx: FLContext):
        """Stop waiting for trainers the server has pruned; they cannot reply (#595).

        A pruned trainer whose result already arrived this round is kept: the
        aggregator has accepted it and it counts. The status dict is replaced,
        not mutated, because ``is_done`` iterates it from the monitor thread.
        """
        with self.lock:
            dropped = [n for n in names if n in self.trainer_statuses and not self.trainer_statuses[n].reply_time]
            if not dropped:
                return []
            self.trainer_statuses = {k: v for k, v in self.trainer_statuses.items() if k not in dropped}
            self.trainers = [t for t in self.trainers if t not in dropped]
            if self.min_responses_required > len(self.trainer_statuses):
                self.min_responses_required = len(self.trainer_statuses)
        self.log_warning(
            fl_ctx,
            f"gatherer for round {self.for_round} no longer waits for pruned trainers {dropped}; "
            f"still expecting {[n for n, t in self.trainer_statuses.items() if not t.reply_time]}",
        )
        return dropped

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
        # Current-round bookkeeping for aggregator take-over (#595).
        self._round_lock = threading.Lock()
        self._round = {}
        self._aggr_replacement = {}
        # How long a failed result send waits for the server to prune the aggregator it was
        # sent to before giving up: the server needs the heartbeat timeout (600 s) plus the
        # dead-client grace (60 s) to deem a site disconnected.
        self.aggregator_death_grace = 900.0

    def process_config(self, fl_ctx: FLContext):
        reply = super().process_config(fl_ctx)
        # The workflow id is known only once the configure task has arrived.
        self.engine.register_aux_message_handler(
            topic=prune_topic(self.workflow_id),
            message_handle_func=self._process_prune_notice,
        )
        return reply

    # ---- aggregator take-over ------------------------------------------------------

    def aggregator_replacement(self, name):
        """The client that took over the current round from pruned aggregator ``name``, or None."""
        with self._round_lock:
            return self._aggr_replacement.get(name)

    def note_permission_granted_by(self, aggr):
        with self._round_lock:
            self._round["granted_by"] = aggr

    def local_permission(self, request, fl_ctx: FLContext):
        """Answer a submission-permission request addressed to this client without a self-message."""
        engine = fl_ctx.get_engine()
        local_fl_ctx = fl_ctx.clone()
        local_fl_ctx.set_peer_context(engine.new_context())
        return self._process_submission_request(self.request_to_submit_learn_result_task_name, request, local_fl_ctx)

    def _gather_locally(self, result, fl_ctx: FLContext):
        """Hand a result to this client's own gatherer, as the stock aggr == self path does."""
        result = self._resolve_lazy_refs(result, fl_ctx)
        engine = fl_ctx.get_engine()
        local_fl_ctx = fl_ctx.clone()
        local_fl_ctx.set_peer_context(engine.new_context())
        return self._process_learn_result(result, local_fl_ctx, fl_ctx.get_run_abort_signal())

    def _handle_pruned_aggregator(self, removed, fl_ctx: FLContext):
        with self._round_lock:
            rd = dict(self._round)
        old_aggr = rd.get("aggr")
        if not rd or old_aggr not in removed:
            return
        candidates = self.get_config_prop(Constant.AGGR_CLIENTS) or self.get_config_prop(Constant.CLIENTS) or []
        if not candidates:
            self.log_error(fl_ctx, f"aggregator {old_aggr} of round {rd.get('num')} was pruned and no candidate remains")
            return
        new_aggr = candidates[0]  # the same choice on every surviving client
        with self._round_lock:
            self._aggr_replacement[old_aggr] = new_aggr
            for k, v in list(self._aggr_replacement.items()):
                if v == old_aggr:
                    self._aggr_replacement[k] = new_aggr
            self._round["aggr"] = new_aggr
        self.log_warning(
            fl_ctx,
            f"FaultTolerant: aggregator {old_aggr} of round {rd.get('num')} was pruned; {new_aggr} takes over the round",
        )
        if new_aggr == self.me:
            self._take_over_round(rd, fl_ctx)
        if rd.get("done") and rd.get("submitted_to") == old_aggr and rd.get("result") is not None:
            # My result was accepted by the dead aggregator and is gone with it.
            threading.Thread(
                target=self._resubmit_round_result,
                args=(rd["result"], rd.get("num"), new_aggr, rd.get("fl_ctx") or fl_ctx),
                name=f"resubmit-round-{rd.get('num')}",
                daemon=True,
            ).start()

    def _take_over_round(self, rd, fl_ctx: FLContext):
        round_num = rd.get("num")
        gatherer = self.gatherer
        if gatherer is not None and getattr(gatherer, "for_round", None) == round_num:
            return
        if gatherer is not None:
            self.log_error(fl_ctx, f"cannot take over round {round_num}: still gathering round {gatherer.for_round}")
            return
        task_data = rd.get("task_data")
        round_fl_ctx = rd.get("fl_ctx") or fl_ctx
        self.log_warning(fl_ctx, f"FaultTolerant: setting up the gatherer for round {round_num} as replacement aggregator")
        self.gatherer = FaultTolerantGatherer(
            fl_ctx=round_fl_ctx,
            all_clients=self.get_config_prop(Constant.CLIENTS),
            metric_comparator=self.metric_comparator,
            trainers=list(self.trainers),
            for_round=round_num,
            timeout=self.learn_task_timeout,
            min_responses_required=self.min_responses_required,
            wait_time_after_min_resps_received=self.wait_time_after_min_resps_received,
            aggregator=self.aggregator,
            executor=self,
            task_data=task_data,
            max_concurrent_submissions=self.max_concurrent_submissions,
        )
        self.gatherer_waiter.set()

    def _resubmit_round_result(self, result, round_num, aggr, fl_ctx: FLContext):
        self.log_warning(fl_ctx, f"FaultTolerant: re-submitting my round {round_num} result to {aggr}")
        reply = self._submit_result_to(aggr, round_num, result, fl_ctx, need_permission=True)
        rc = _PermissionReplyRetryEngine._return_code(reply)
        if rc == ReturnCode.OK:
            with self._round_lock:
                self._round["submitted_to"] = aggr
            self.log_info(fl_ctx, f"round {round_num} result accepted by replacement aggregator {aggr}")
        else:
            self.log_error(fl_ctx, f"re-submission of round {round_num} result to {aggr} failed: {rc}")

    def _submit_result_to(self, aggr, round_num, result, fl_ctx: FLContext, need_permission: bool):
        """Permission loop plus result send to ``aggr`` (local when that is this client)."""
        if need_permission:
            req = Shareable()
            req.set_header(AppConstants.CURRENT_ROUND, round_num)
            max_wait = self.request_to_submit_result_max_wait or 0
            start = time.time()
            abort = fl_ctx.get_run_abort_signal()
            while True:
                if abort is not None and getattr(abort, "triggered", False):
                    return None
                if aggr == self.me:
                    reply = self.local_permission(req, fl_ctx)
                else:
                    resp = fl_ctx.get_engine().send_aux_request(
                        targets=[aggr],
                        topic=self.request_to_submit_learn_result_task_name,
                        request=req,
                        timeout=self.request_to_submit_result_msg_timeout,
                        fl_ctx=fl_ctx,
                        secure=False,
                    )
                    reply = resp.get(aggr) if isinstance(resp, dict) else None
                rc = _PermissionReplyRetryEngine._return_code(reply)
                if rc == ReturnCode.OK:
                    break
                if rc == ReturnCode.MODEL_UNRECOGNIZED:
                    return reply
                if max_wait and time.time() - start > max_wait:
                    return reply
                later = self.aggregator_replacement(aggr)
                if later and later != aggr:
                    aggr = later  # the replacement died too; follow the next election
                time.sleep(self.request_to_submit_result_interval)
        if aggr == self.me:
            return self._gather_locally(result, fl_ctx)
        task = Task(
            name=self.report_learn_result_task_name,
            data=result,
            timeout=int(self.learn_task_ack_timeout),
            secure=self.is_task_secure(fl_ctx),
        )
        resp = SwarmClientController.broadcast_and_wait(self, task, fl_ctx, [aggr], 1)
        return resp.get(aggr) if isinstance(resp, dict) else None

    def _wait_for_replacement(self, orig, fl_ctx: FLContext):
        """A result send to ``orig`` failed. Give the server time to prune it; return its replacement or None."""
        deadline = time.time() + self.aggregator_death_grace
        abort = fl_ctx.get_run_abort_signal()
        self.log_warning(
            fl_ctx,
            f"result send to aggregator {orig} failed; waiting up to {self.aggregator_death_grace:.0f}s for the server "
            f"to prune it before giving up",
        )
        while time.time() < deadline:
            if abort is not None and getattr(abort, "triggered", False):
                return None
            repl = self.aggregator_replacement(orig)
            if repl and repl != orig:
                return repl
            time.sleep(5)
        return None

    def broadcast_and_wait(self, task, fl_ctx: FLContext, targets=None, min_responses=1,
                           wait_time_after_min_received=0, abort_signal=None):
        """Redirect the learn-result send when the round's aggregator has been pruned (#595)."""
        if getattr(task, "name", None) != self.report_learn_result_task_name or not targets or len(targets) != 1:
            return super().broadcast_and_wait(task, fl_ctx, targets, min_responses, wait_time_after_min_received, abort_signal)
        orig = targets[0]
        with self._round_lock:
            self._round["result"] = task.data
            self._round["submitted_to"] = orig
            round_num = self._round.get("num")
            granted_by = self._round.get("granted_by")
        repl = self.aggregator_replacement(orig)
        if repl and repl != orig:
            reply = self._submit_result_to(repl, round_num, task.data, fl_ctx, need_permission=(granted_by != repl))
            return self._finish_send(orig, repl, reply)
        resp = super().broadcast_and_wait(task, fl_ctx, targets, min_responses, wait_time_after_min_received, abort_signal)
        reply = resp.get(orig) if isinstance(resp, dict) else None
        if _PermissionReplyRetryEngine._return_code(reply) == ReturnCode.OK:
            return self._finish_send(orig, orig, reply)
        repl = self._wait_for_replacement(orig, fl_ctx)
        if not repl:
            return resp
        reply = self._submit_result_to(repl, round_num, task.data, fl_ctx, need_permission=True)
        return self._finish_send(orig, repl, reply)

    def _finish_send(self, orig, actual, reply):
        if _PermissionReplyRetryEngine._return_code(reply) == ReturnCode.OK:
            with self._round_lock:
                self._round["submitted_to"] = actual
                self._round["done"] = True
        return {orig: reply}

    def _process_prune_notice(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        pruned = list(request.get(PRUNE_KEY_PRUNED) or [])
        active = list(request.get(PRUNE_KEY_ACTIVE) or [])
        reason = request.get(PRUNE_KEY_REASON, "?")
        self.apply_prune(pruned, active, reason, fl_ctx)
        return make_reply(ReturnCode.OK)

    def apply_prune(self, pruned, active, reason, fl_ctx: FLContext):
        """Forget pruned clients everywhere a later round would still use them (#595):
        the scatter targets (TRAIN_CLIENTS), the aggregator draw (AGGR_CLIENTS), the
        final-result recipients (RESULT_CLIENTS), and a gatherer waiting on them now."""
        removed = set(pruned)
        if not removed:
            return
        if self.me in removed:
            # The server thinks we are gone (e.g. a long disconnect). Nothing sensible
            # to do here but say so; the server ignores our reports from now on.
            self.log_error(fl_ctx, f"server pruned THIS client ({reason}); remaining active clients {active}")
            return
        if not self.config:
            self.log_warning(fl_ctx, f"prune notice for {pruned} arrived before configuration; ignored")
            return
        for key in (Constant.CLIENTS, Constant.TRAIN_CLIENTS, Constant.AGGR_CLIENTS, Constant.RESULT_CLIENTS):
            names = self.config.get(key)
            if names:
                self.config[key] = [c for c in names if c not in removed]
        if self.trainers:
            self.trainers = [c for c in self.trainers if c not in removed]
        aggrs = getattr(self, "aggrs", None)
        if aggrs:
            self.aggrs = [c for c in aggrs if c not in removed]
        gatherer = self.gatherer
        if gatherer is not None and hasattr(gatherer, "drop_trainers"):
            gatherer.drop_trainers(removed, fl_ctx)
        self.log_warning(
            fl_ctx,
            f"FaultTolerant: server pruned {sorted(removed)} ({reason}); this swarm now trains on "
            f"{self.get_config_prop(Constant.TRAIN_CLIENTS)} and aggregates on "
            f"{self.get_config_prop(Constant.AGGR_CLIENTS)}",
        )
        self._handle_pruned_aggregator(removed, fl_ctx)

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
        with self._round_lock:
            self._round = {
                "num": task_data.get_header(AppConstants.CURRENT_ROUND),
                "aggr": task_data.get_header(Constant.AGGREGATOR),
                "task_data": task_data,
                "fl_ctx": retry_fl_ctx,
                "result": None,
                "submitted_to": None,
                "granted_by": None,
                "done": False,
            }
            self._aggr_replacement = {}
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
    """SwarmServerController that prunes a failed client and continues, instead of
    panicking the whole run, as long as ``min_clients`` remains -- and tells the
    surviving clients about it (#595).

    Three things prune a client: an error report (``_update_client_status``, a
    faithful copy of the stock method with only that branch changed), the server
    deeming it disconnected (heartbeat timeout plus grace; a reconnect within that
    window never reaches here), and not answering the configure task before the
    quorum was reached (the task completes at quorum, so it never will).

    In strict mode (``min_clients`` = 0 or = site count) nothing is pruned: a site
    that loses its VPN keeps training locally and submits when the tunnel returns,
    and the strict gatherer waits for it. Pruning there would turn a survivable
    outage into a failed benchmark, so a disconnect is only logged."""

    def __init__(self, *args, prune_notify_timeout: float = 30.0, **kwargs):
        if prune_notify_timeout <= 0:
            raise ValueError("prune_notify_timeout must be positive")
        super().__init__(*args, **kwargs)
        self.prune_notify_timeout = prune_notify_timeout
        self.pruned_clients = []

    def _active_clients(self):
        return list(self.client_statuses.keys())

    def _can_prune_one(self) -> bool:
        remaining = len(self.client_statuses) - 1
        return bool(self.min_clients and self.min_clients > 0 and remaining >= self.min_clients)

    def _prune(self, names, reason: str, fl_ctx: FLContext):
        """Drop ``names`` from the active set and notify the survivors. Callers check min_clients."""
        names = [n for n in names if n in self.client_statuses]
        if not names:
            return
        for n in names:
            del self.client_statuses[n]
            self.pruned_clients.append(n)
        self.log_warning(
            fl_ctx,
            f"FaultTolerant: pruned {names} ({reason}); {len(self.client_statuses)} active clients remain "
            f"(min_clients={self.min_clients}): {self._active_clients()}",
        )
        self._notify_prune(names, reason, fl_ctx)

    def _notify_prune(self, pruned, reason: str, fl_ctx: FLContext):
        active = self._active_clients()
        if not active:
            return
        engine = fl_ctx.get_engine()
        if not engine:
            self.log_error(fl_ctx, f"no engine: cannot tell {active} that {pruned} were pruned")
            return
        request = Shareable()
        request[PRUNE_KEY_PRUNED] = list(pruned)
        request[PRUNE_KEY_ACTIVE] = active
        request[PRUNE_KEY_REASON] = reason
        try:
            replies = engine.send_aux_request(
                targets=active,
                topic=prune_topic(self.workflow_id),
                request=request,
                timeout=self.prune_notify_timeout,
                fl_ctx=fl_ctx,
                secure=False,
            )
        except Exception:
            self.log_error(fl_ctx, f"error notifying {active} of pruned clients {pruned}: {secure_format_traceback()}")
            return
        replies = replies or {}
        acked = [c for c in active if _LearnScatterRetryEngine._return_code(replies.get(c)) == ReturnCode.OK]
        missing = [c for c in active if c not in acked]
        if missing:
            self.log_warning(
                fl_ctx,
                f"prune notice for {pruned} acknowledged by {acked}; no acknowledgement from {missing} "
                f"within {self.prune_notify_timeout}s -- they may still scatter to or wait for {pruned}",
            )
        else:
            self.log_info(fl_ctx, f"prune notice for {pruned} acknowledged by all active clients {acked}")

    def _configure_clients(self, learn_config, fl_ctx: FLContext, abort_signal) -> bool:
        if not super()._configure_clients(learn_config, fl_ctx, abort_signal):
            return False
        unconfigured = [c for c in self._active_clients() if not self._is_configured(c)]
        if unconfigured:
            # The quorum proceeded without them and the configure task is complete, so
            # they will never be configured. Before the start task goes out, make the
            # roster the starting client will scatter to match reality (#595 B).
            self._prune(unconfigured, "not configured when the quorum was reached", fl_ctx)
        return True

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)
        if event_type == EventType.CLIENT_DISCONNECTED:
            self._on_client_disconnected(fl_ctx.get_prop(FLContextKey.DISCONNECTED_CLIENT_NAME), fl_ctx)
        elif event_type == EventType.CLIENT_RECONNECTED:
            name = fl_ctx.get_prop(FLContextKey.RECONNECTED_CLIENT_NAME)
            if name in self.pruned_clients:
                self.log_warning(fl_ctx, f"client {name} reconnected but stays pruned for this run (rejoin is not supported)")

    def _on_client_disconnected(self, client_name, fl_ctx: FLContext):
        if not client_name or client_name not in self.client_statuses:
            return
        if self._can_prune_one():
            self._prune([client_name], "deemed disconnected by the server", fl_ctx)
            return
        self.log_warning(
            fl_ctx,
            f"client {client_name} is deemed disconnected; not pruned (min_clients={self.min_clients}, "
            f"active={len(self.client_statuses)}) -- the run waits for it to reconnect and submit",
        )

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
                self._prune([client_name], f"error report '{report.error}'", fl_ctx)
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
