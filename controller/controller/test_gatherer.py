import unittest
import time
import logging
from unittest.mock import MagicMock

from gatherer import _TrainerStatus, Gatherer

from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.ccwf.client_ctl import ClientSideController
from nvflare.app_common.ccwf.common import Constant
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.metric_comparator import MetricComparator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestGatherer(unittest.TestCase):
    CLIENT_THAT_TRAINS = 'client_a'
    CLIENT_THAT_DOES_NOT_TRAIN = 'client_b'
    def setUp(self):
        # TODO think about if this makes sense as a generic setup or if different tests require different setup
        self.fl_context = FLContext()
        self.aggregator = MagicMock(Aggregator)
        self.gatherer = Gatherer(task_data = MagicMock(Shareable),
                                 fl_ctx = self.fl_context,
                                 for_round = 0,
                                 executor = MagicMock(ClientSideController),
                                 aggregator = self.aggregator,
                                 metric_comparator = MagicMock(MetricComparator),
                                 all_clients = [self.CLIENT_THAT_TRAINS, self.CLIENT_THAT_DOES_NOT_TRAIN],
                                 trainers =  [self.CLIENT_THAT_TRAINS],
                                 min_responses_required = 1,
                                 wait_time_after_min_resps_received = 1,
                                 timeout = 1)

    def test_trainer_status_can_be_accessed(self):
        name = "MyName"
        now = time.time()
        trainer_status = _TrainerStatus(name)
        trainer_status.reply_time = now
        self.assertEqual(name, trainer_status.name)
        self.assertEqual(now, trainer_status.reply_time)

    def test_default_gatherer_is_not_done(self):
        print("This test does not work yet.")
        return
        # TODO use one of the following lines, depending on whether a return value of None is OK, if it needs to be False, etc.
        #      self.assertFalse(self.gatherer.is_done())
        #      self.assertEqual(self.gatherer.is_done(), False)  # self.assertFalse(arg) only checks for "not arg"
        #      or something else if a dummy Gatherer is expected to be done

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
                self.gatherer = Gatherer(task_data=task_data,
                                         fl_ctx=self.fl_context,
                                         for_round=0,
                                         executor=MagicMock(ClientSideController),
                                         aggregator=self.aggregator,
                                         metric_comparator=MagicMock(MetricComparator),
                                         all_clients=[self.CLIENT_THAT_TRAINS, self.CLIENT_THAT_DOES_NOT_TRAIN],
                                         trainers=[self.CLIENT_THAT_TRAINS],
                                         min_responses_required=1,
                                         wait_time_after_min_resps_received=1,
                                         timeout=1)
            self.assertEqual(log.output, [expected_message])

    def test_gatherer_is_done(self):
        print("This test is not implemented yet.")
        # TODO Think about which different scenarios to test when Gatherer is done and when it is not.
        #      We probably at least have the cases
        #        ‣ all finished → is done
        #        ‣ timeout
        #        ‣ minimum number not received after grace period, exactly minimum number received, more than minimum number received [not sure I understood the logic correctly]

    def test_gathering_todo_expected_behavior(self):
        print("This test is not implemented yet.")
        # TODO Think about which different scenarios to test, what the expected behavior is, and how to implement these tests (probably as multiple methods, probably by setting up real objects rather than mocking (too much to mock)?)
        #      We probably at least have the cases
        #        ‣ valid result from correct round from trainer
        #        ‣ received result from non-trainer
        #        ‣ received early result from trainer
        #        ‣ received late  result from trainer
        #        ‣ received bad result
        #        ‣ exception thrown
        #      and may want to check in each case
        #        ‣ that correct information was logged
        #        ‣ return values
        #        ‣ exceptions (in case an exception is expected, the test for non-occurring exceptions is implicit)
        #        ‣ state of the gatherer
        #        ‣ event fired?

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
