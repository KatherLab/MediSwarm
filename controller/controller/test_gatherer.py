import unittest
import time
import logging
from mock import mock
from numpy import NaN

from gatherer import _TrainerStatus, Gatherer

from nvflare.apis.signal import Signal
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.client_ctl import ClientSideController
from nvflare.app_common.ccwf.common import Constant, NumberMetricComparator
from nvflare.app_common.abstract.aggregator import Aggregator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MockedResult(Shareable):
    def __init__(self, current_round: int):
        super().__init__()
        self.current_round: int = current_round

    def get_header(self, key):
        if key == AppConstants.CURRENT_ROUND:
            return self.current_round

    def get_return_code(self, _):
        return ReturnCode.OK

class MockedResultRaisingException(MockedResult):
    def get_header(self, _):
        raise Exception("Test exception")

class MockedResultFailing(MockedResult):
    def get_return_code(self, _):
        return ReturnCode.EXECUTION_RESULT_ERROR


class MockedAggregator(Aggregator):
    def accept(self, _a: Shareable, _b: FLContext) -> bool:
        return True

    def aggregate(self, _: FLContext) -> Shareable:
        return Shareable()

class MockedAggregatorRaisingException(MockedAggregator):
    def aggregate(self, _: FLContext) -> Shareable:
        raise Exception("foo")

class MockedClientSideController(ClientSideController):
    def __init__(self):
        super().__init__(task_name_prefix="test_prefix")

    def do_learn_task(self, _a: str, _b: Shareable, _c: FLContext, _d: Signal):
        pass

    def start_workflow(self, _a: Shareable, _b: FLContext, _c: Signal) -> Shareable:
        pass

class EventCatcher():
    def __init__(self):
        self.events_caught = []

    def catch_event(self, *event_args):
        self.events_caught.append(event_args[0])

class TestGatherer(unittest.TestCase):
    CLIENT_THAT_TRAINS = 'client_a'
    CLIENT_THAT_DOES_NOT_TRAIN = 'client_b'
    OTHER_CLIENT_THAT_TRAINS = 'client_c'

    def _get_gatherer(self,
                      task_data = Shareable(),
                      for_round = 0,
                      all_clients = [CLIENT_THAT_TRAINS, CLIENT_THAT_DOES_NOT_TRAIN],
                      trainers = [CLIENT_THAT_TRAINS],
                      min_responses_required = 1,
                      ):
        return Gatherer(task_data = task_data,
                        fl_ctx = self.fl_context,
                        for_round = for_round,
                        executor = MockedClientSideController(),
                        aggregator = self.aggregator,
                        metric_comparator = NumberMetricComparator(),
                        all_clients = all_clients,
                        trainers =  trainers,
                        min_responses_required = min_responses_required,
                        wait_time_after_min_resps_received = 0.1,
                        timeout = 0.1)

    def setUp(self):
        self.fl_context = FLContext()
        self.aggregator = MockedAggregator()
        self.gatherer = self._get_gatherer()

    def test_trainer_status_can_be_accessed(self):
        name = "test name"
        now = time.time()
        trainer_status = _TrainerStatus(name)
        trainer_status.reply_time = now
        self.assertEqual(name, trainer_status.name)
        self.assertEqual(now, trainer_status.reply_time)


    def test_gatherer_initialization_logs_correctly(self):
        for current_best_client, expected_message in ((None, "INFO:Gatherer:[identity=, run=?]: gatherer starting from scratch"),
                                                      (self.CLIENT_THAT_TRAINS, "INFO:Gatherer:[identity=, run=?]: gatherer starting with previous best result from client client_a with metric None at round None")):
            with (self.assertLogs(logging.getLogger("Gatherer"), logging.INFO) as log):
                task_data = Shareable()
                task_data.set_header(Constant.CLIENT, current_best_client)
                self.gatherer = self._get_gatherer(task_data=task_data)
            self.assertEqual(log.output, [expected_message])

    def test_gatherer_returns_error_on_result_from_non_training_client(self):
        result = MockedResult(0)
        with self.assertLogs(logging.getLogger("Gatherer"), logging.ERROR) as log:
            response = self.gatherer.gather(self.CLIENT_THAT_DOES_NOT_TRAIN, result, self.fl_context)
        self.assertEqual(make_reply(ReturnCode.EXECUTION_EXCEPTION), response)
        expected_message = f"ERROR:Gatherer:[identity=, run=?]: Received result from {self.CLIENT_THAT_DOES_NOT_TRAIN} for round 0, but it is not a trainer"
        self.assertIn(expected_message, log.output)

    def test_gatherer_receives_from_earlier_round_logs_warning(self):
        current_round = 2
        self.gatherer = self._get_gatherer(for_round=current_round)
        result = MockedResult(current_round - 1)
        with self.assertLogs(logging.getLogger("Gatherer"), logging.INFO) as log:
            self.gatherer.gather(self.CLIENT_THAT_TRAINS, result, self.fl_context)
        expected_message = f"WARNING:Gatherer:[identity=, run=?]: Received late result from {self.CLIENT_THAT_TRAINS} for round {current_round-1}, which is < gatherer's current round {current_round}"
        self.assertIn(expected_message, log.output)

    def test_gatherer_receives_from_later_round_logs_warning(self):
        current_round = 1
        self.gatherer = self._get_gatherer(for_round=current_round)
        result = MockedResult(current_round + 1)
        with self.assertLogs(logging.getLogger("Gatherer"), logging.ERROR) as log:
            response = self.gatherer.gather(self.CLIENT_THAT_TRAINS, result, self.fl_context)
        self.assertEqual(make_reply(ReturnCode.EXECUTION_EXCEPTION), response)
        expected_message = f"ERROR:Gatherer:[identity=, run=?]: Logic error: received result from {self.CLIENT_THAT_TRAINS} for round {current_round+1}, which is > gatherer's current round {current_round}"
        self.assertIn(expected_message, log.output)

    def test_gatherer_logs_exception_from_gathering(self):
        result = MockedResultRaisingException(0)
        with self.assertLogs(logging.getLogger("Gatherer"), logging.ERROR) as log:  # but does not raise exception
            self.gatherer.gather(self.CLIENT_THAT_TRAINS, result, self.fl_context)
        self.assertTrue(log.output[0].startswith("ERROR:Gatherer:[identity=, run=?]: Exception gathering"))

    def test_gatherer_gathering_from_current_round_with_enough_responses_gets_logged_and_events_are_fired(self):
        event_catcher = EventCatcher()

        current_round = 0
        result = MockedResult(current_round)
        with self.assertLogs(logging.getLogger("Gatherer"), logging.INFO) as log, \
             mock.patch('nvflare.apis.fl_component.FLComponent.fire_event', side_effect=event_catcher.catch_event):
            response = self.gatherer.gather(self.CLIENT_THAT_TRAINS, result, self.fl_context)
        self.assertEqual(make_reply(ReturnCode.OK), response)
        expected_entry = f"INFO:Gatherer:[identity=, run=?]: Contribution from {self.CLIENT_THAT_TRAINS} ACCEPTED by the aggregator at round 0."
        self.assertTrue(expected_entry in log.output)
        self.assertListEqual(event_catcher.events_caught, ['_before_contribution_accept', '_after_contribution_accept'])

    def test_gatherer_gathering_bad_result_gets_logged(self):
        current_round = 0
        result = MockedResultFailing(current_round)
        with self.assertLogs(logging.getLogger("Gatherer"), logging.ERROR) as log:
            response = self.gatherer.gather(self.CLIENT_THAT_TRAINS, result, self.fl_context)
        self.assertEqual(make_reply(ReturnCode.EXECUTION_EXCEPTION), response)
        expected_entry = f"ERROR:Gatherer:[identity=, run=?]: Bad result from {self.CLIENT_THAT_TRAINS} for round {current_round}: EXECUTION_RESULT_ERROR."
        self.assertTrue(expected_entry in log.output)

    def _set_metrics(self, executor_best: float, current_best: float):
        self.gatherer.executor.best_metric = executor_best
        self.gatherer.current_best_global_metric = current_best

    def test_aggregating_returns_error_on_exception_during_aggregation(self):
        self.gatherer.aggregator = MockedAggregatorRaisingException()
        self._set_metrics(0.0, 0.0)
        self.assertEqual(make_reply(ReturnCode.EXECUTION_EXCEPTION), self.gatherer.aggregate())

    def _prepare_for_aggregation(self, executor_best: float, current_best: float):
        current_round = 0
        result = MockedResult(current_round)
        self.gatherer = self._get_gatherer()
        self._set_metrics(executor_best, current_best)
        self.gatherer.gather(self.CLIENT_THAT_TRAINS, result, self.fl_context)

    def test_aggregating_determines_best_metric_correctly(self):
        for executor_best, current_best, best, first_is_better in ((0.4,  0.6,  0.6, False ), # other is better (note: for metrics, larger is better)
                                                                   (0.6,  0.4,  0.6, True  ), # own is better
                                                                   (0.5,  0.5,  0.5, False ), # own is not better, take other
                                                                   (None, 0.5,  0.5, False ), # 0.5 is better than None
                                                                   (0.5,  None, 0.5, True  ),
                                                                   (0.5,  NaN,  0.5, True  ),
                                                                   (NaN,  0.5,  0.5, False )):
            self._prepare_for_aggregation(executor_best, current_best)
            with self.assertLogs(logging.getLogger("Gatherer"), logging.INFO) as log:
                result = self.gatherer.aggregate()

            if first_is_better:
                self.assertTrue("INFO:Gatherer:[identity=, run=?]: Finished aggregation for round 0" == log.output[-3])
                self.assertTrue(log.output[-2].startswith("INFO:Gatherer:[identity=, run=?]: I got better metric"))
            else:
                self.assertTrue("INFO:Gatherer:[identity=, run=?]: Finished aggregation for round 0" == log.output[-2])
            self.assertAlmostEqual(best, result.get_header(Constant.METRIC))

    def test_aggregating_fires_events(self):
        event_catcher = EventCatcher()

        self._prepare_for_aggregation(0.4, 0.6)
        with mock.patch('nvflare.apis.fl_component.FLComponent.fire_event', side_effect=event_catcher.catch_event):
            self.gatherer.aggregate()

        self.assertListEqual(event_catcher.events_caught, ['_before_aggregation', '_after_aggregation'])

    def test_gatherer_is_done_if_all_are_finished(self):
        for trainer in self.gatherer.trainer_statuses.keys():
            self.gatherer.trainer_statuses[trainer].reply_time = time.time()
        self.assertTrue(self.gatherer.is_done())

    def test_gatherer_is_done_if_timeout(self):
        # TODO Does that make sense (i.e., is there a check whether something _useful_ has been gathered later in the workflow?)
        #      Does the context have the information why/in which way the gatherer is done?
        time.sleep(0.11)
        with self.assertLogs(logging.getLogger("Gatherer"), logging.INFO) as log:
            self.assertTrue(self.gatherer.is_done())
        self.assertTrue("WARNING:Gatherer:[identity=, run=?]: Gatherer for round 0 timed out after 0.1 seconds" in log.output)

    def test_gatherer_is_done_if_enough_responses_received(self):
        time.sleep(0.11)
        self.gatherer = self._get_gatherer(all_clients=[self.CLIENT_THAT_TRAINS, self.OTHER_CLIENT_THAT_TRAINS],
                                           trainers=[self.CLIENT_THAT_TRAINS, self.OTHER_CLIENT_THAT_TRAINS],
                                           min_responses_required=1)
        self.gatherer.trainer_statuses[self.OTHER_CLIENT_THAT_TRAINS].reply_time = time.time()
        print("This test does not work yet")
        # TODO After timeout, having received the minimum number of responses, should the gatherer report being done (it doesn’t) and log so (it doesn’t)
        # with self.assertLogs(logging.getLogger("Gatherer"), logging.INFO) as log:
        #     self.assertTrue(self.gatherer.is_done())
        # self.assertTrue("WARNING:Gatherer:[identity=, run=?]: Gatherer for round 0 exit after 0.1 seconds since received minimum responses" in log.output)

    def test_gatherer_is_not_done_if_no_trainer_is_finished(self):
        self.assertIsNone(self.gatherer.is_done())

    def test_gatherer_is_not_done_if_insufficient_responses_received(self):
        time.sleep(0.11)
        self.gatherer = self._get_gatherer(all_clients=[self.CLIENT_THAT_TRAINS, self.OTHER_CLIENT_THAT_TRAINS],
                                           trainers=[self.CLIENT_THAT_TRAINS, self.OTHER_CLIENT_THAT_TRAINS],
                                           min_responses_required=2)
        self.gatherer.trainer_statuses[self.OTHER_CLIENT_THAT_TRAINS].reply_time = time.time()
        self.assertIsNone(self.gatherer.is_done())
