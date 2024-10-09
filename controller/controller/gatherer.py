import threading
import time
import numpy as np

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.metric_comparator import MetricComparator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.client_ctl import ClientSideController
from nvflare.app_common.ccwf.common import Constant
from nvflare.security.logging import secure_format_traceback


class _TrainerStatus:
    """
    Internal class to keep track of trainer's status including reply time.
    """
    def __init__(self, name: str):
        self.name = name
        self.reply_time = None


class Gatherer(FLComponent):
    """
    Gatherer class responsible for gathering and aggregating training results from multiple clients
    during the swarm learning process.
    """
    def __init__(
        self,
        task_data: Shareable,
        fl_ctx: FLContext,
        for_round: int,
        executor: ClientSideController,
        aggregator: Aggregator,
        metric_comparator: MetricComparator,
        all_clients: list,
        trainers: list,
        min_responses_required: int,
        wait_time_after_min_resps_received: float,
        timeout,
    ):
        super().__init__()
        self.fl_ctx = fl_ctx
        self.executor = executor
        self.aggregator = aggregator
        self.metric_comparator = metric_comparator
        self.all_clients = all_clients
        self.trainers = trainers
        self.for_round = for_round
        self.trainer_statuses = {}
        self.start_time = time.time()
        self.timeout = timeout

        for t in trainers:
            self.trainer_statuses[t] = _TrainerStatus(t)
        if min_responses_required <= 0 or min_responses_required >= len(trainers):
            min_responses_required = len(trainers)
        self.min_responses_required = min_responses_required
        self.wait_time_after_min_resps_received = wait_time_after_min_resps_received
        self.min_resps_received_time = None
        self.lock = threading.Lock()
        self.current_best_client = task_data.get_header(Constant.CLIENT)
        self.current_best_global_metric = task_data.get_header(Constant.METRIC)
        self.current_best_round = task_data.get_header(Constant.ROUND)
        if not self.current_best_client:
            self.log_info(fl_ctx, "gatherer starting from scratch")
        else:
            self.log_info(
                fl_ctx,
                f"gatherer starting with previous best result from client {self.current_best_client} "
                f"with metric {self.current_best_global_metric} "
                f"at round {self.current_best_round}",
            )

    def gather(self, client_name: str, result: Shareable, fl_ctx: FLContext) -> Shareable:
        """
        Gather the results from a client and perform aggregation if applicable.
        """
        with self.lock:
            try:
                return self._do_gather(client_name, result, fl_ctx)
            except Exception as e:
                self.log_error(fl_ctx, f"Exception gathering: {secure_format_traceback()}")
                self.log_error(fl_ctx, f"Exception during gather: {e}")  # TODO need to log twice?
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _do_gather(self, client_name: str, result: Shareable, fl_ctx: FLContext) -> Shareable:
        result_round = result.get_header(AppConstants.CURRENT_ROUND)
        ts = self.trainer_statuses.get(client_name)
        if not ts:
            self.log_error(
                fl_ctx, f"Received result from {client_name} for round {result_round}, but it is not a trainer"
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if result_round > self.for_round:
            self.log_error(
                fl_ctx,
                f"Logic error: received result from {client_name} for round {result_round}, "
                f"which is > gatherer's current round {self.for_round}",
            )
            self.executor.update_status(action="gather", error=ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if result_round < self.for_round:
            self.log_warning(
                fl_ctx,
                f"Received late result from {client_name} for round {result_round}, "
                f"which is < gatherer's current round {self.for_round}",
            )

        if result_round == self.for_round:
            now = time.time()
            ts.reply_time = now
            if not self.min_resps_received_time:
                num_resps_received = sum(1 for ts in self.trainer_statuses.values() if ts.reply_time)
                if num_resps_received >= self.min_responses_required:
                    self.min_resps_received_time = now

        rc = result.get_return_code(ReturnCode.OK)
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"Bad result from {client_name} for round {result_round}: {rc}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

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

    def aggregate(self):
        """
        Perform the aggregation of results gathered from trainers.
        """

        def _is_valid_value(value: float) -> bool:
            return ( value is not None ) and ( not np.isnan(value) )

        fl_ctx = self.fl_ctx
        self.log_info(fl_ctx, f"Start aggregation for round {self.for_round}")
        self.fire_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
        try:
            aggr_result = self.aggregator.aggregate(fl_ctx)
        except Exception as e:
            self.log_error(fl_ctx, f"Exception in aggregation: {secure_format_traceback()}")
            self.log_error(fl_ctx, f"Exception during aggregation: {e}")  # TODO need to log twice?
            self.executor.update_status(action="aggregate", error=ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
        self.fire_event(AppEventType.AFTER_AGGREGATION, fl_ctx)
        self.log_info(fl_ctx, f"Finished aggregation for round {self.for_round}")

        mine_is_better = False
        if _is_valid_value(self.current_best_global_metric):
            if (
                _is_valid_value(self.executor.best_metric)
                and self.metric_comparator.compare(self.executor.best_metric, self.current_best_global_metric) > 0
            ):
                mine_is_better = True
        elif _is_valid_value(self.executor.best_metric):
            mine_is_better = True

        # Determine if the local metric is better than the current global best
        if mine_is_better:
            self.log_info(
                fl_ctx, f"I got better metric {self.executor.best_metric} at round {self.executor.best_round}"
            )
            best_round = self.executor.best_round
            best_metric = self.executor.best_metric
            best_client = self.executor.me
        else:
            best_round = self.current_best_round
            best_metric = self.current_best_global_metric
            best_client = self.current_best_client

        self.log_info(fl_ctx, f"Global best metric is {best_metric} from client {best_client} at round {best_round}")

        aggr_result.set_header(Constant.ROUND, best_round)
        aggr_result.set_header(Constant.METRIC, best_metric)
        aggr_result.set_header(Constant.CLIENT, best_client)

        return aggr_result

    def is_done(self):
        """
        Check if the gather process is complete, either by receiving all responses or timing out.
        For compatibility with NVFlare Gatherer, do not return False, but None.
        """
        unfinished = sum(1 for s in self.trainer_statuses.values() if not s.reply_time)
        if unfinished == 0:
            return True

        now = time.time()
        if self.timeout and now - self.start_time > self.timeout:
            self.log_warning(self.fl_ctx, f"Gatherer for round {self.for_round} timed out after {self.timeout} seconds")
            return True

        if self.min_resps_received_time and now - self.min_resps_received_time > self.wait_time_after_min_resps_received:
            self.log_info(
                self.fl_ctx,
                f"Gatherer for round {self.for_round} exit after {self.wait_time_after_min_resps_received} seconds "
                f"since received minimum responses",
            )
            return True
