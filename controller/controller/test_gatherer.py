import unittest
import time
import logging
from unittest.mock import MagicMock

from gatherer import _TrainerStatus, Gatherer

from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.client_ctl import ClientSideController
from nvflare.app_common.ccwf.common import Constant
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.metric_comparator import MetricComparator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MockedResultForTesting(Shareable):
    def __init__(self, current_round: int):
        super().__init__()
        self.current_round: int = current_round

    def get_header(self, key):
        if key == AppConstants.CURRENT_ROUND:
            return self.current_round

    def get_return_code(self, _):
        return ReturnCode.OK

class MockedResultRaisingExceptionForTesting(MockedResultForTesting):
    def get_header(self, _):
        raise Exception("Test exception")

class MockedResultFailingForTesting(MockedResultForTesting):
    def get_return_code(self, _):
        return ReturnCode.EXECUTION_RESULT_ERROR

class TestGatherer(unittest.TestCase):
    CLIENT_THAT_TRAINS = 'client_a'
    CLIENT_THAT_DOES_NOT_TRAIN = 'client_b'
    OTHER_CLIENT_THAT_TRAINS = 'client_c'

    def _get_gatherer(self,
                      task_data = MagicMock(Shareable),
                      for_round = 0,
                      all_clients = [CLIENT_THAT_TRAINS, CLIENT_THAT_DOES_NOT_TRAIN],
                      trainers = [CLIENT_THAT_TRAINS],
                      min_responses_required = 1,
                      ):
        return Gatherer(task_data = task_data,
                        fl_ctx = self.fl_context,
                        for_round = for_round,
                        executor = MagicMock(ClientSideController),
                        aggregator = self.aggregator,
                        metric_comparator = MagicMock(MetricComparator),
                        all_clients = all_clients,
                        trainers =  trainers,
                        min_responses_required = min_responses_required,
                        wait_time_after_min_resps_received = 1,
                        timeout = 1)

    def setUp(self):
        self.fl_context = FLContext()
        self.aggregator = MagicMock(Aggregator)
        self.gatherer = self._get_gatherer()


    def test_trainer_status_can_be_accessed(self):
        name = "test name"
        now = time.time()
        trainer_status = _TrainerStatus(name)
        trainer_status.reply_time = now
        self.assertEqual(name, trainer_status.name)
        self.assertEqual(now, trainer_status.reply_time)


    def test_gatherer_initialization_logs_correctly(self):
        class TestTaskData(Shareable):
            def __init__(self, current_best_client):
                self._current_best_client = current_best_client
                super().__init__()

            def get_header(self, key: str, default=None) -> bool:
                if key == Constant.CLIENT:
                    return self._current_best_client
                else:
                    return super().get_header(key)

        for current_best_client, expected_message in ((None, "INFO:Gatherer:[identity=, run=?]: gatherer starting from scratch"),
                                                      (self.CLIENT_THAT_TRAINS,  "INFO:Gatherer:[identity=, run=?]: gatherer starting with previous best result from client client_a with metric None at round None")):  # TODO is this trailing whitespace intended?
            with self.assertLogs(logging.getLogger("Gatherer"), logging.INFO) as log:
                task_data = TestTaskData(current_best_client=current_best_client)
                self.gatherer = self._get_gatherer(task_data=task_data)
            self.assertEqual(log.output, [expected_message])

    def test_gatherer_returns_error_on_result_from_non_training_client(self):
        result = MockedResultForTesting(0)
        with self.assertLogs(logging.getLogger("Gatherer"), logging.ERROR) as log:
            response = self.gatherer.gather(self.CLIENT_THAT_DOES_NOT_TRAIN, result, self.fl_context)
        self.assertEqual(make_reply(ReturnCode.EXECUTION_EXCEPTION), response)
        expected_message = f"ERROR:Gatherer:[identity=, run=?]: Received result from {self.CLIENT_THAT_DOES_NOT_TRAIN} for round 0, but it is not a trainer"
        self.assertIn(expected_message, log.output)

    def test_gatherer_receives_from_earlier_round_logs_warning(self):
        current_round = 2
        self.gatherer = self._get_gatherer(for_round=current_round)
        result = MockedResultForTesting(current_round-1)
        with self.assertLogs(logging.getLogger("Gatherer"), logging.INFO) as log:
            self.gatherer.gather(self.CLIENT_THAT_TRAINS, result, self.fl_context)
        expected_message = f"WARNING:Gatherer:[identity=, run=?]: Received late result from {self.CLIENT_THAT_TRAINS} for round {current_round-1}, which is < gatherer's current round {current_round}"
        self.assertIn(expected_message, log.output)

    def test_gatherer_receives_from_later_round_logs_warning(self):
        current_round = 1
        self.gatherer = self._get_gatherer(for_round=current_round)
        result = MockedResultForTesting(current_round+1)
        with self.assertLogs(logging.getLogger("Gatherer"), logging.ERROR) as log:
            response = self.gatherer.gather(self.CLIENT_THAT_TRAINS, result, self.fl_context)
        self.assertEqual(make_reply(ReturnCode.EXECUTION_EXCEPTION), response)
        expected_message = f"ERROR:Gatherer:[identity=, run=?]: Logic error: received result from {self.CLIENT_THAT_TRAINS} for round {current_round+1}, which is > gatherer's current round {current_round}"
        self.assertIn(expected_message, log.output)

    def test_gatherer_logs_exception_from_gathering(self):
        result = MockedResultRaisingExceptionForTesting(0)
        with self.assertLogs(logging.getLogger("Gatherer"), logging.ERROR) as log:  # but does not raise exception
            self.gatherer.gather(self.CLIENT_THAT_TRAINS, result, self.fl_context)
        self.assertTrue(log.output[0].startswith("ERROR:Gatherer:[identity=, run=?]: Exception gathering"))

    def test_gatherer_gathering_from_current_round_without_enough_responses_TODO(self):
        current_round = 0
        self.gatherer = self._get_gatherer(for_round=current_round,
                                           all_clients=[self.CLIENT_THAT_TRAINS, self.OTHER_CLIENT_THAT_TRAINS],
                                           trainers=[self.CLIENT_THAT_TRAINS, self.OTHER_CLIENT_THAT_TRAINS],
                                           min_responses_required=2)
        result = MockedResultForTesting(current_round)
        response = self.gatherer.gather(self.CLIENT_THAT_TRAINS, result, self.fl_context)
        # TODO what is expected behavior if there are not enough responses?
        print("This test does not work yet.")

    def test_gatherer_gathering_from_current_round_with_enough_responses_gets_logged(self):
        current_round = 0
        result = MockedResultForTesting(current_round)
        with self.assertLogs(logging.getLogger("Gatherer"), logging.INFO) as log:
            response = self.gatherer.gather(self.CLIENT_THAT_TRAINS, result, self.fl_context)
        self.assertEqual(make_reply(ReturnCode.OK), response)
        expected_entry = f"INFO:Gatherer:[identity=, run=?]: Contribution from {self.CLIENT_THAT_TRAINS} ACCEPTED by the aggregator at round 0."
        self.assertTrue(expected_entry in log.output)

    def test_gatherer_gathering_bad_result_gets_logged(self):
        current_round = 0
        result = MockedResultFailingForTesting(current_round)
        with self.assertLogs(logging.getLogger("Gatherer"), logging.ERROR) as log:
            response = self.gatherer.gather(self.CLIENT_THAT_TRAINS, result, self.fl_context)
        self.assertEqual(make_reply(ReturnCode.EXECUTION_EXCEPTION), response)
        expected_entry = f"ERROR:Gatherer:[identity=, run=?]: Bad result from {self.CLIENT_THAT_TRAINS} for round {current_round}: EXECUTION_RESULT_ERROR."
        self.assertTrue(expected_entry in log.output)


    def test_aggregating_returns_error_on_exception_during_aggregation(self):
        self.aggregator.aggregate.side_effect=Exception("foo")
        self.assertEqual(make_reply(ReturnCode.EXECUTION_EXCEPTION), self.gatherer.aggregate())
        # TODO is this the correct behavior: exception is caught, error code is returned and we could continue?

    def test_aggregating_todo_expected_behavior(self):
        print("This test is not implemented yet.")
        # TODO Think about which different scenarios to test, what the expected behavior is, and how to implement these tests (probably as multiple methods, probably by setting up real objects rather than mocking (too much to mock)?)
        #      We probably at least have the cases
        #        ‣ best result from this client
        #        ‣ best result from other client
        #        ‣ tie (equally good results) from other clients
        #        ‣ tie between this and other client
        #        ‣ Do we need to consider cases of NaN, Inf, zero, negative results?
        #        ‣ exception thrown
        #      and may want to check in each case
        #        ‣ that correct information was logged
        #        ‣ return values
        #        ‣ exceptions (in case an exception is expected, the test for non-occurring exceptions is implicit)
        #        ‣ state of the gatherer
        #        ‣ event fired?

    def test_default_gatherer_is_not_done(self):
        print("This test does not work yet.")
        return
        # TODO use one of the following lines, depending on whether a return value of None is OK, if it needs to be False, etc.
        #      self.assertFalse(self.gatherer.is_done())
        #      self.assertEqual(self.gatherer.is_done(), False)  # self.assertFalse(arg) only checks for "not arg"
        #      or something else if a dummy Gatherer is expected to be done

    def test_gatherer_is_done(self):
        print("This test is not implemented yet.")
        # TODO Think about which different scenarios to test when Gatherer is done and when it is not.
        #      Actually, how can a gatherer be "not done"?
        #      We probably at least have the cases
        #        ‣ all finished → is done
        #        ‣ timeout
        #        ‣ minimum number not received after grace period, exactly minimum number received, more than minimum number received [not sure I understood the logic correctly]

# TODO
# ‣ Can we test that events were fired?
