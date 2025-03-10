import copy
import logging
import random
import threading
import time

from controller.gatherer import Gatherer

from nvflare.apis.controller_spec import Task
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.metric_comparator import MetricComparator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.client_ctl import ClientSideController
from nvflare.app_common.ccwf.common import Constant, NumberMetricComparator, ResultType, make_task_name
from nvflare.fuel.utils.validation_utils import check_non_empty_str, check_positive_int, check_positive_number
from nvflare.security.logging import secure_format_traceback

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
            self.log_info(fl_ctx, "Gatherer starting from scratch")
        else:
            self.log_info(
                fl_ctx,
                f"Gatherer starting with previous best result from client {self.current_best_client} "
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
                logger.error(f"Exception during gather: {e}")
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

    def aggregate(self) -> Shareable:
        """
        Perform the aggregation of results gathered from trainers.
        """
        fl_ctx = self.fl_ctx
        self.log_info(fl_ctx, f"Start aggregation for round {self.for_round}")
        self.fire_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
        try:
            aggr_result = self.aggregator.aggregate(fl_ctx)
        except Exception as e:
            self.log_error(fl_ctx, f"Exception in aggregation: {secure_format_traceback()}")
            logger.error(f"Exception during aggregation: {e}")
            self.executor.update_status(action="aggregate", error=ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
        self.fire_event(AppEventType.AFTER_AGGREGATION, fl_ctx)
        self.log_info(fl_ctx, f"Finished aggregation for round {self.for_round}")

        # Determine if the local metric is better than the current global best
        mine_is_better = False
        if self.current_best_global_metric is not None:
            if (
                self.executor.best_metric is not None
                and self.metric_comparator.compare(self.executor.best_metric, self.current_best_global_metric) > 0
            ):
                mine_is_better = True
        elif self.executor.best_metric is not None:
            mine_is_better = True

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

    def is_done(self) -> bool:
        """
        Check if the gather process is complete, either by receiving all responses or timing out.
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
                f"Gatherer for round {self.for_round} exiting after {self.wait_time_after_min_resps_received} seconds "
                f"since receiving minimum responses",
            )
            return True


class SwarmClientController(ClientSideController):
    """
    The SwarmClientController class manages the client-side execution of the swarm learning workflow.
    It handles the training, aggregation, and communication with other clients in a decentralized manner.
    """
    def __init__(
        self,
        task_name_prefix=Constant.TN_PREFIX_SWARM,  # Prefix for tasks associated with the swarm workflow
        learn_task_name=AppConstants.TASK_TRAIN,  # Name of the task to be executed for learning
        persistor_id=AppConstants.DEFAULT_PERSISTOR_ID,  # ID of the persistor component
        shareable_generator_id=AppConstants.DEFAULT_SHAREABLE_GENERATOR_ID,  # ID of the shareable generator component
        aggregator_id=AppConstants.DEFAULT_AGGREGATOR_ID,  # ID of the aggregator component
        metric_comparator_id=None,  # Optional ID for a custom metric comparator
        learn_task_check_interval=Constant.LEARN_TASK_CHECK_INTERVAL,  # Interval for checking learning tasks (in seconds)
        learn_task_abort_timeout=Constant.LEARN_TASK_ABORT_TIMEOUT,  # Timeout for aborting a learning task (in seconds)
        learn_task_ack_timeout=Constant.LEARN_TASK_ACK_TIMEOUT,  # Timeout for acknowledging a learning task (in seconds)
        learn_task_timeout=None,  # Timeout for the overall learning task (in seconds)
        final_result_ack_timeout=Constant.FINAL_RESULT_ACK_TIMEOUT,  # Timeout for acknowledging the final result (in seconds)
        min_responses_required: int = 1,  # Minimum number of responses required to proceed
        wait_time_after_min_resps_received: float = 10.0,  # Time to wait after minimum responses are received (in seconds)
    ):
        """
        Initializes the SwarmClientController, validating the input parameters and setting up internal state.
        """
        try:
            # Validate required arguments
            check_non_empty_str("learn_task_name", learn_task_name)
            check_non_empty_str("persistor_id", persistor_id)
            check_non_empty_str("shareable_generator_id", shareable_generator_id)
            check_non_empty_str("aggregator_id", aggregator_id)

            if metric_comparator_id is not None:
                check_non_empty_str("metric_comparator_id", metric_comparator_id)

            if learn_task_timeout is not None:
                check_positive_number("learn_task_timeout", learn_task_timeout)

            check_positive_int("min_responses_required", min_responses_required)
            check_positive_number("wait_time_after_min_resps_received", wait_time_after_min_resps_received)

            super().__init__(
                task_name_prefix=task_name_prefix,
                learn_task_name=learn_task_name,
                persistor_id=persistor_id,
                shareable_generator_id=shareable_generator_id,
                learn_task_check_interval=learn_task_check_interval,
                learn_task_ack_timeout=learn_task_ack_timeout,
                learn_task_abort_timeout=learn_task_abort_timeout,
                final_result_ack_timeout=final_result_ack_timeout,
                allow_busy_task=True,
            )
            # Initialize internal variables
            self.metric_comparator_id = metric_comparator_id
            self.metric_comparator = None
            self.report_learn_result_task_name = make_task_name(task_name_prefix, Constant.BASENAME_REPORT_LEARN_RESULT)
            self.learn_task_timeout = learn_task_timeout
            self.min_responses_required = min_responses_required
            self.wait_time_after_min_resps_received = wait_time_after_min_resps_received
            self.aggregator_id = aggregator_id
            self.aggregator = None
            self.gatherer = None
            self.gatherer_waiter = threading.Event()
            self.trainers = None
            self.aggrs = None
            self.is_trainer = False
            self.is_aggr = False
            self.last_aggr_round_done = -1
        except Exception as e:
            logging.getLogger("SwarmClientController").log(logging.ERROR, f"Error during initialization: {e}")
            # cannot log via self.log_error because we have no FLContext here
            raise

    def process_config(self, fl_ctx: FLContext):
        """
        Process the configuration for the swarm learning workflow, identifying the roles of the current client.
        """
        try:
            all_clients = self.get_config_prop(Constant.CLIENTS)

            self.trainers = self.get_config_prop(Constant.TRAIN_CLIENTS)
            if not self.trainers:
                self.trainers = all_clients
            self.is_trainer = (self.me in self.trainers)

            self.aggrs = self.get_config_prop(Constant.AGGR_CLIENTS)
            if not self.aggrs:
                self.aggrs = all_clients
            self.is_aggr = (self.me in self.aggrs)

            # Register message handler for sharing results
            self.engine.register_aux_message_handler(
                topic=self.topic_for_my_workflow(Constant.TOPIC_SHARE_RESULT),
                message_handle_func=self._process_share_result,
            )
        except Exception as e:
            self.log_error(fl_ctx, f"Exception during process_config: {secure_format_traceback()}")
            raise

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """
        Execute a specific task based on the task name. Handles both regular and learning result tasks.
        """
        try:
            if task_name == self.report_learn_result_task_name:
                return self._process_learn_result(shareable, fl_ctx, abort_signal)
            return super().execute(task_name, shareable, fl_ctx, abort_signal)
        except Exception as e:
            self.log_error(fl_ctx, f"Exception during execute: {secure_format_traceback()}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def start_run(self, fl_ctx: FLContext):
        """
        Start the swarm learning run, setting up the aggregator and metric comparator components.
        """
        try:
            super().start_run(fl_ctx)
            self.aggregator = self.engine.get_component(self.aggregator_id)
            if not isinstance(self.aggregator, Aggregator):
                self.system_panic(
                    f"Aggregator {self.aggregator_id} must be an Aggregator but got {type(self.aggregator)}",
                    fl_ctx,
                )
                return

            if self.metric_comparator_id:
                self.metric_comparator = self.engine.get_component(self.metric_comparator_id)
                if not isinstance(self.metric_comparator, MetricComparator):
                    self.system_panic(
                        f"Metric comparator {self.metric_comparator_id} must be a MetricComparator "
                        f"but got {type(self.metric_comparator)}",
                        fl_ctx,
                    )
                    return
            else:
                self.metric_comparator = NumberMetricComparator()

            # Start a thread to monitor the gather process
            aggr_thread = threading.Thread(target=self._monitor_gather)
            aggr_thread.daemon = True
            aggr_thread.start()
            self.log_info(fl_ctx, "Started aggregator thread")
        except Exception as e:
            self.log_error(fl_ctx, f"Exception during start_run: {secure_format_traceback()}")
            raise

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """
        Handle specific events, such as when a global best model is available, updating the client's status.
        """
        try:
            if event_type == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
                client = fl_ctx.get_prop(Constant.CLIENT)
                if client and client != self.me:
                    return

                self.best_metric = fl_ctx.get_prop(AppConstants.VALIDATION_RESULT)
                self.best_result = copy.deepcopy(fl_ctx.get_prop(AppConstants.GLOBAL_MODEL))
                self.log_info(fl_ctx, f"Got GLOBAL_BEST_MODEL_AVAILABLE: best metric={self.best_metric}")
                current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
                self.best_round = current_round
                self.update_status(last_round=current_round, action="better_aggregation")
            else:
                super().handle_event(event_type, fl_ctx)
        except Exception as e:
            self.log_error(fl_ctx, f"Exception during handle_event: {secure_format_traceback()}")
            raise

    def start_workflow(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """
        Start the swarm learning workflow by scattering tasks to the appropriate clients.
        """
        try:
            clients = self.get_config_prop(Constant.CLIENTS)
            aggr_clients = self.get_config_prop(Constant.AGGR_CLIENTS, [])
            train_clients = self.get_config_prop(Constant.TRAIN_CLIENTS, [])

            self.log_info(
                fl_ctx, f"Starting Swarm Workflow on clients {clients}, aggrs {aggr_clients}, trainers {train_clients}"
            )

            if not self._scatter(
                task_data=shareable, for_round=self.get_config_prop(Constant.START_ROUND, 0), fl_ctx=fl_ctx
            ):
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            self.log_info(fl_ctx, "Started Swarm Workflow")
            return make_reply(ReturnCode.OK)
        except Exception as e:
            self.log_error(fl_ctx, f"Exception during start_workflow: {secure_format_traceback()}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _scatter(self, task_data: Shareable, for_round: int, fl_ctx: FLContext) -> bool:
        """
        Distribute learning tasks to the training and aggregation clients for a specific round.
        """
        try:
            clients = self.get_config_prop(Constant.TRAIN_CLIENTS)
            aggr_clients = self.get_config_prop(Constant.AGGR_CLIENTS)

            aggr = random.choice(aggr_clients)

            task_data.set_header(AppConstants.CURRENT_ROUND, for_round)
            task_data.add_cookie(AppConstants.CONTRIBUTION_ROUND, for_round)
            task_data.set_header(Constant.AGGREGATOR, aggr)

            targets = copy.copy(clients)
            if aggr not in targets:
                targets.append(aggr)

            self.log_info(fl_ctx, f"Broadcasting learn task of round {for_round} to {targets}; aggr client is {aggr}")
            return self.send_learn_task(targets=targets, request=task_data, fl_ctx=fl_ctx)
        except Exception as e:
            self.log_error(fl_ctx, f"Exception during _scatter: {secure_format_traceback()}")
            return False

    def _monitor_gather(self):
        """
        Monitor the gather process to check if the aggregation for a round is complete.
        """
        while True:
            if self.asked_to_stop:
                return

            gatherer = self.gatherer
            if gatherer:
                assert isinstance(gatherer, Gatherer)
                if gatherer.is_done():
                    self.last_aggr_round_done = gatherer.for_round
                    self.gatherer = None
                    self.gatherer_waiter.clear()
                    try:
                        self._end_gather(gatherer)
                    except Exception as e:
                        self.log_error(gatherer.fl_ctx, f"Exception ending gatherer: {secure_format_traceback()}")
                        self.update_status(action="aggregate", error=ReturnCode.EXECUTION_EXCEPTION)
            time.sleep(0.2)

    def _end_gather(self, gatherer: Gatherer):
        """
        Finalize the aggregation process and determine the next steps in the workflow.
        """
        fl_ctx = gatherer.fl_ctx
        try:
            aggr_result = gatherer.aggregate()
        except Exception as e:
            self.log_error(fl_ctx, f"Exception in aggregation: {secure_format_traceback()}")
            self.update_status(action="aggregate", error=ReturnCode.EXECUTION_EXCEPTION)
            return

        self.log_debug(fl_ctx, f"Aggregation result: {aggr_result}")
        global_weights = self.shareable_generator.shareable_to_learnable(aggr_result, fl_ctx)
        self.record_last_result(fl_ctx, gatherer.for_round, global_weights)

        num_rounds_done = gatherer.for_round - self.get_config_prop(Constant.START_ROUND, 0) + 1
        if num_rounds_done >= self.get_config_prop(AppConstants.NUM_ROUNDS):
            self.log_info(fl_ctx, f"Swarm Learning Done: number of rounds completed {num_rounds_done}")

            self._distribute_final_results(aggr_result, fl_ctx)
            return

        next_round_data = self.shareable_generator.learnable_to_shareable(global_weights, fl_ctx)
        assert isinstance(next_round_data, Shareable)

        best_round = aggr_result.get_header(Constant.ROUND)
        best_metric = aggr_result.get_header(Constant.METRIC)
        best_client = aggr_result.get_header(Constant.CLIENT)

        if best_client:
            next_round_data.set_header(Constant.ROUND, best_round)
            next_round_data.set_header(Constant.CLIENT, best_client)
            next_round_data.set_header(Constant.METRIC, best_metric)

        self._scatter(next_round_data, gatherer.for_round + 1, gatherer.fl_ctx)

    def _ask_to_share_best_result(self, client: str, metric, fl_ctx: FLContext):
        """
        Request the client with the best metric to share its result with the other clients.
        """
        try:
            self.log_info(fl_ctx, f"Client {client} has the best metric {metric} - asking it to share result")
            resp = self.engine.send_aux_request(
                targets=[client],
                topic=self.topic_for_my_workflow(Constant.TOPIC_SHARE_RESULT),
                request=Shareable(),
                timeout=self.final_result_ack_timeout,
                fl_ctx=fl_ctx,
                secure=False,
            )

            assert isinstance(resp, dict)
            reply = resp.get(client)
            if not reply:
                self.log_error(fl_ctx, f"Failed to ask client {client} to share final result")
                return

            if not isinstance(reply, Shareable):
                self.log_error(fl_ctx, f"Client {client} failed to respond to share final result request")
                return

            rc = reply.get_return_code()
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"Client {client} failed to respond to share final result request: {rc}")
        except Exception as e:
            self.log_error(fl_ctx, f"Exception during _ask_to_share_best_result: {secure_format_traceback()}")

    def _distribute_final_results(self, aggr_result: Shareable, fl_ctx: FLContext):
        """
        Distribute the final results of the swarm learning process to all clients.
        """
        try:
            best_client = aggr_result.get_header(Constant.CLIENT)
            best_metric = aggr_result.get_header(Constant.METRIC)

            if best_client:
                if best_client == self.me:
                    self.log_info(fl_ctx, f"I have the global best metric {best_metric}")
                    self.broadcast_final_result(
                        fl_ctx, ResultType.BEST, self.best_result, self.best_metric, self.best_round
                    )
                else:
                    self._ask_to_share_best_result(best_client, best_metric, fl_ctx)
            else:
                self.log_info(fl_ctx, "No global best result!")

            self.log_info(fl_ctx, "Distributing last result")
            self.broadcast_final_result(fl_ctx, ResultType.LAST, self.last_result, round_num=self.last_round)
        except Exception as e:
            self.log_error(fl_ctx, f"Exception during _distribute_final_results: {secure_format_traceback()}")

    def _process_learn_result(self, request: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """
        Process the learning result received from a peer client.
        """
        try:
            peer_ctx = fl_ctx.get_peer_context()
            assert isinstance(peer_ctx, FLContext)
            client_name = peer_ctx.get_identity_name()
            current_round = request.get_header(AppConstants.CURRENT_ROUND)
            self.log_info(fl_ctx, f"Got training result from {client_name} for round {current_round}")

            peer_ctx.set_prop(FLContextKey.SHAREABLE, request)

            gatherer = self.gatherer
            if not gatherer:
                if current_round <= self.last_aggr_round_done:
                    self.log_info(fl_ctx, f"Dropped result from late {client_name} for round {current_round}")
                    return make_reply(ReturnCode.OK)

                self.log_info(fl_ctx, f"Got result from {client_name} for round {current_round} before gatherer setup")
                self.gatherer_waiter.wait(self.learn_task_abort_timeout)

            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            gatherer = self.gatherer
            if not gatherer:
                self.log_error(fl_ctx, f"Still no gatherer after {self.learn_task_abort_timeout} seconds")
                self.log_error(fl_ctx, f"Ignored result from {client_name} for round {current_round} since no gatherer")
                self.update_status(action="wait_for_gatherer", error=ReturnCode.EXECUTION_EXCEPTION)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            assert isinstance(gatherer, Gatherer)
            if gatherer.for_round != current_round:
                self.log_warning(
                    fl_ctx,
                    f"Got result from {client_name} for round {current_round}, "
                    f"but I'm waiting for round {gatherer.for_round}",
                )
            return gatherer.gather(client_name, request, fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception processing learn result: {secure_format_traceback()}")
            self.log_error(fl_ctx, f"Exception processing learn result: {e}")
            self.update_status(action="process_learn_result", error=ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def do_learn_task(self, name: str, task_data: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        """
        Perform the learning task as part of the swarm learning workflow, handling the training and communication
        with the aggregator.
        """
        try:
            current_round = task_data.get_header(AppConstants.CURRENT_ROUND)
            self.update_status(last_round=current_round, action="start_learn_task")

            aggr = task_data.get_header(Constant.AGGREGATOR)
            if not aggr:
                self.log_error(fl_ctx, f"Missing aggregation client for round {current_round}")
                self.update_status(action="do_learn_task", error=ReturnCode.EXECUTION_EXCEPTION)
                return

            self.log_info(fl_ctx, f"Round {current_round} started.")
            task_data.set_header(FLContextKey.TASK_NAME, name)

            base_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
            if not base_model:
                base_model = Learnable()
                fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, base_model, private=True, sticky=True)
            global_weights = self.shareable_generator.shareable_to_learnable(task_data, fl_ctx)

            self.log_debug(fl_ctx, f"Current global model: {global_weights}")

            fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, global_weights, private=True, sticky=True)
            fl_ctx.set_prop(AppConstants.CURRENT_ROUND, current_round, private=True, sticky=True)
            self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)

            if self.me == aggr:
                gatherer = self.gatherer
                if gatherer:
                    self.log_error(
                        fl_ctx,
                        f"Logic error: got task for round {current_round} while gathering for round {gatherer.for_round}",
                    )
                    self.update_status(action="do_learn_task", error=ReturnCode.EXECUTION_EXCEPTION)
                    return

                self.log_info(fl_ctx, f"Setting up the gatherer for round {current_round}")

                self.gatherer = Gatherer(
                    fl_ctx=fl_ctx,
                    all_clients=self.get_config_prop(Constant.CLIENTS),
                    metric_comparator=self.metric_comparator,
                    trainers=self.trainers,
                    for_round=current_round,
                    timeout=self.learn_task_timeout,
                    min_responses_required=self.min_responses_required,
                    wait_time_after_min_resps_received=self.wait_time_after_min_resps_received,
                    aggregator=self.aggregator,
                    executor=self,
                    task_data=task_data,
                )
                self.gatherer_waiter.set()

            if self.is_trainer:
                result = self.execute_learn_task(task_data, fl_ctx, abort_signal)

                rc = result.get_return_code(ReturnCode.OK)
                if rc != ReturnCode.OK:
                    self.log_error(fl_ctx, f"Learn executor failed: {rc}")
                    self.update_status(action="learner_execution", error=rc)
                    return

                self.log_info(fl_ctx, f"Sending training result to aggregation client {aggr}")
                task = Task(
                    name=self.report_learn_result_task_name,
                    data=result,
                    timeout=int(self.learn_task_ack_timeout),
                    secure=self.is_task_secure(fl_ctx),
                )

                resp = self.broadcast_and_wait(
                    task=task,
                    targets=[aggr],
                    min_responses=1,
                    fl_ctx=fl_ctx,
                )

                reply = resp.get(aggr)
                if not reply:
                    self.log_error(fl_ctx, f"Failed to receive reply from aggregation client: {aggr}")
                    self.update_status(action="receive_learn_result_reply", error=ReturnCode.EXECUTION_EXCEPTION)
                    return

                if not isinstance(reply, Shareable):
                    self.log_error(
                        fl_ctx, f"Bad reply from aggregation client {aggr}: expect Shareable but got {type(reply)}"
                    )
                    self.update_status(action="receive_learn_result_reply", error=ReturnCode.EXECUTION_EXCEPTION)
                    return

                rc = reply.get_return_code(ReturnCode.OK)
                if rc != ReturnCode.OK:
                    self.log_error(fl_ctx, f"Bad return code from aggregation client {aggr}: {rc}")
                    self.update_status(action="receive_learn_result_reply", error=ReturnCode.EXECUTION_EXCEPTION)
                    return

                self.log_info(fl_ctx, f"Finished round {current_round}")
                self.update_status(last_round=current_round, action="finished_learn_task")
        except Exception as e:
            self.log_error(fl_ctx, f"Exception during do_learn_task: {secure_format_traceback()}")
            self.update_status(action="do_learn_task", error=ReturnCode.EXECUTION_EXCEPTION)

    def _process_share_result(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """
        Process a request from another client to share the current best result.
        """
        try:
            peer_ctx = fl_ctx.get_peer_context()
            assert isinstance(peer_ctx, FLContext)
            client_name = peer_ctx.get_identity_name()
            if not self.best_result:
                self.log_error(
                    fl_ctx, f"Got request from {client_name} to share my best result, but I don't have a best result"
                )
                return make_reply(ReturnCode.BAD_REQUEST_DATA)

            self.update_status(action="start_share_result_request_process")
            self.broadcast_final_result(
                fl_ctx, ResultType.BEST, self.best_result, metric=self.best_metric, round_num=self.best_round
            )
            return make_reply(ReturnCode.OK)
        except Exception as e:
            self.log_error(fl_ctx, f"Exception during _process_share_result: {secure_format_traceback()}")
            self.log_error(fl_ctx, f"Exception during _process_share_result: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

# Test Cases

import unittest
from unittest.mock import MagicMock

class TestSwarmClientController(unittest.TestCase):

    def setUp(self):
        """
        Set up a mock FLContext and instantiate the SwarmClientController with test data
        for unit testing.
        """
        self.fl_ctx = MagicMock(FLContext)
        self.controller = SwarmClientController(
            task_name_prefix="test_prefix",
            learn_task_name="test_learn_task",
            persistor_id="test_persistor",
            shareable_generator_id="test_generator",
            aggregator_id="test_aggregator"
        )

    def test_initialization(self):
        """
        Test the initialization of the SwarmClientController to ensure proper assignment
        of attributes.
        """
        self.assertIsInstance(self.controller, SwarmClientController)
        self.assertEqual(self.controller.task_name_prefix, "test_prefix")
        self.assertEqual(self.controller.learn_task_name, "test_learn_task")

    def test_process_config(self):
        """
        Test the process_config method to verify correct role assignment as trainer or aggregator.
        """
        self.controller.me = "client1"
        self.controller.get_config_prop = MagicMock(side_effect=lambda x, y=None: ["client1", "client2", "client3"])
        self.controller.process_config(self.fl_ctx)
        self.assertTrue(self.controller.is_trainer)
        self.assertTrue(self.controller.is_aggr)

    def test_execute(self):
        """
        Test the execute method to ensure proper handling of the task execution flow.
        """
        shareable = MagicMock(Shareable)
        abort_signal = MagicMock(Signal)
        self.controller._process_learn_result = MagicMock(return_value=shareable)
        result = self.controller.execute("test_learn_task", shareable, self.fl_ctx, abort_signal)
        self.assertEqual(result, shareable)

    def test_start_run(self):
        """
        Test the start_run method to verify that components such as the aggregator are set up correctly.
        """
        self.controller.engine = MagicMock()
        self.controller.engine.get_component = MagicMock(return_value=MagicMock(Aggregator))
        self.controller.start_run(self.fl_ctx)
        self.assertIsInstance(self.controller.aggregator, Aggregator)

    def test_handle_event(self):
        """
        Test the handle_event method to check if global best model events are handled correctly.
        """
        self.controller.me = "client1"
        self.fl_ctx.get_prop = MagicMock(return_value="client1")
        self.fl_ctx.set_prop = MagicMock()
        self.controller.best_result = None
        self.controller.handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, self.fl_ctx)
        self.assertIsNotNone(self.controller.best_result)

    def test_start_workflow(self):
        """
        Test the start_workflow method to ensure the workflow is initiated correctly.
        """
        shareable = MagicMock(Shareable)
        abort_signal = MagicMock(Signal)
        self.controller._scatter = MagicMock(return_value=True)
        result = self.controller.start_workflow(shareable, self.fl_ctx, abort_signal)
        self.assertEqual(result.get_return_code(), ReturnCode.OK)

if __name__ == "__main__":
    unittest.main()
