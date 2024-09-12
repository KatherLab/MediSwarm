import unittest
import logging
from unittest.mock import MagicMock

from swarm_client_ctl import SwarmClientController
from nvflare.apis.fl_context import FLContext

# TODO move constants to suitable location
TASK_NAME_PREFIX = 'test_prefix'
LEARN_TASK_NAME = 'test_learn_task'

class TestSwarmClientController(unittest.TestCase):

    def setup_controller(self,
                        task_name_prefix=TASK_NAME_PREFIX,
                        learn_task_name=LEARN_TASK_NAME,
                        persistor_id="test_persistor_id",
                        shareable_generator_id="test_generator_id",
                        aggregator_id="test_aggregator_id",
                        **kwargs
    ):
        self.controller = SwarmClientController(
            task_name_prefix=task_name_prefix,
            learn_task_name=learn_task_name,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            aggregator_id=aggregator_id,
            **kwargs
        )

    def setUp(self):
        """
        Set up a mock FLContext and instantiate the SwarmClientController with test data
        for unit testing.
        """
        self.fl_ctx = MagicMock(FLContext)
        self.controller = None
        self.setup_controller()
        self.testee_logger = logging.getLogger("swarm_client_ctl")

    def test_initialization_sets_members_correctly(self):
        """
        Test the initialization of the SwarmClientController to ensure proper assignment
        of attributes.
        """
        self.assertIsInstance(self.controller, SwarmClientController)
        self.assertEqual(self.controller.task_name_prefix, TASK_NAME_PREFIX)
        self.assertEqual(self.controller.learn_task_name, LEARN_TASK_NAME)

    def test_incorrect_initializations_raise_errors_and_logs(self):
        print("This test is not complete yet.")
        for argument_empty in ('learn_task_name','persistor_id', 'shareable_generator_id', 'aggregator_id'):
            # TODO should also fail on empty 'metric_comparator_id' (but does not)
            with self.assertLogs(self.testee_logger, logging.ERROR) as log, self.assertRaises(ValueError) as error:
                self.setup_controller(**{argument_empty: ''})
            self.assertTrue(log.output[0].startswith(f"ERROR:swarm_client_ctl:Error during initialization: {argument_empty} must not be empty"))

        print("This test is not complete yet.")
        for nonpositive_number, value in (('learn_task_timeout', -1.0), ('min_responses_required', 0), ('wait_time_after_min_resps_received', 0.0)):
            # no need to distinguish between float and int here
            # TODO should also fail for learn_task_timeout 0.0 (but does not)
            with self.assertLogs(self.testee_logger, logging.ERROR) as log, self.assertRaises(ValueError) as error:
                self.setup_controller(**{nonpositive_number: value})
            self.assertTrue(log.output[0].startswith(f"ERROR:swarm_client_ctl:Error during initialization: {nonpositive_number} must > 0, but got {value}"))


    def test_process_config_sets_client_roles_correctly(self):
        """
        Test the process_config method to verify correct role assignment as trainer or aggregator.
        """
        print("This test does not work yet.")
        # This is prepared and called via execute().
        # TODO think about whether a minimal preparation makes sense for a separate test or whether the method should implicitly tested via execute()
        """
        # TODO cases to check: default (all train, all aggregate), trainers specified (not all), aggregators specified (not all)
        # self.controller.get_config_prop = MagicMock(side_effect=lambda x, y=None: ["client1", "client2", "client3"])
        # self.controller.process_config(self.fl_ctx)
        # self.assertTrue(self.controller.is_trainer)
        # self.assertTrue(self.controller.is_aggr)
        """
        return


    def test_process_config_raises_errors_and_logs(self):
        fl_context = FLContext()
        with self.assertLogs(self.testee_logger, logging.ERROR) as log, self.assertRaises(TypeError) as error:
            self.controller.process_config(fl_context)
        self.assertEqual(log.output[0], "ERROR:swarm_client_ctl:Exception during process_config: argument of type 'NoneType' is not iterable")


    def test_execute(self):
        """
        Test the execute method to ensure proper handling of the task execution flow.
        """
        print("This test does not work yet.")
        return
        shareable = MagicMock(Shareable)
        abort_signal = MagicMock(Signal)
        self.controller._process_learn_result = MagicMock(return_value=shareable)
        result = self.controller.execute("test_learn_task", shareable, self.fl_ctx, abort_signal)
        self.assertEqual(result, shareable)

    def test_start_run(self):
        """
        Test the start_run method to verify that components such as the aggregator are set up correctly.
        """
        print("This test does not work yet.")
        return
        # TODO clarify what needs to be done starting a run is possible
        #      check succcessful execution and logging if self.metric_comparator_id is set
        #                                                                              not set
        #            system_panic if aggregator is not an instance of Aggregator
        #                            metric_comparator is not an instance of MetricComparator
        self.controller.engine = MagicMock()
        self.controller.engine.get_component = MagicMock(return_value=MagicMock(Aggregator))
        self.controller.start_run(self.fl_ctx)
        self.assertIsInstance(self.controller.aggregator, Aggregator)

    def test_handle_event(self):
        """
        Test the handle_event method to check if global best model events are handled correctly.
        """
        print("This test does not work yet.")
        return
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
        print("This test does not work yet.")
        return
        shareable = MagicMock(Shareable)
        abort_signal = MagicMock(Signal)
        self.controller._scatter = MagicMock(return_value=True)
        result = self.controller.start_workflow(shareable, self.fl_ctx, abort_signal)
        self.assertEqual(result.get_return_code(), ReturnCode.OK)