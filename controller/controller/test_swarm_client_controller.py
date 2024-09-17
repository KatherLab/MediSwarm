import unittest
import logging
from mock import mock
from unittest.mock import MagicMock

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.common import Constant

from swarm_client_ctl import SwarmClientController


# TODO move constants to suitable location
TASK_NAME_PREFIX = 'test_prefix'
LEARN_TASK_NAME = 'test_learn_task'


class MockedEngineForTesting:
    # TODO consider merging this with test_swarm_server_controller.MockedEngineForTesting
    def __init__(self):
        self.fl_ctx_mgr = FLContextManager(engine=self)

    def new_context(self):
        context = self.fl_ctx_mgr.new_context()
        return context

    def register_aux_message_handler(self, topic, message_handle_func):
        pass

    def fire_event(self, event_type, fl_ctx):
        pass

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
        self.controller = None
        self.setup_controller()
        self.engine = None
        self.fl_context = None
        self.testee_logger = logging.getLogger("swarm_client_ctl")

    def test_initialization_sets_members_correctly(self):
        """
        Test the initialization of the SwarmClientController to ensure proper assignment
        of attributes.
        """
        self.assertIsInstance(self.controller, SwarmClientController)
        self.assertEqual(self.controller.task_name_prefix, TASK_NAME_PREFIX)
        self.assertEqual(self.controller.learn_task_name, LEARN_TASK_NAME)
        metric_comparator_id = "test_metric_comparator_id"
        self.setup_controller(metric_comparator_id=metric_comparator_id)
        self.assertEqual(self.controller.metric_comparator_id, metric_comparator_id)

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

    def _setup_for_processing_config(self, config):
        self.setup_controller()
        fl_engine = MockedEngineForTesting()
        fl_context = fl_engine.new_context()
        self.controller.me = config[Constant.CLIENTS][0]
        self.controller.engine = fl_engine
        self.controller.config = config
        return fl_context

    def test_process_config_sets_client_roles_correctly(self):
        """
        Test the process_config method to verify correct role assignment as trainer or aggregator.
        """
        for config in ({Constant.CLIENTS: ['C1', 'C2', 'C3'], Constant.TRAIN_CLIENTS: ['C1', 'C2'], Constant.AGGR_CLIENTS: ['C2', 'C3']},
                       {Constant.CLIENTS: ['C1', 'C2', 'C3'], Constant.TRAIN_CLIENTS: ['C2', 'C3'], Constant.AGGR_CLIENTS: ['C1', 'C3']},
                       {Constant.CLIENTS: ['C1', 'C2', 'C3'], Constant.TRAIN_CLIENTS: ['C1', 'C2'], Constant.AGGR_CLIENTS: ['C1', 'C3']},
                       {Constant.CLIENTS: ['C1', 'C2', 'C3'], Constant.TRAIN_CLIENTS: ['C1', 'C2']                                     },
                       {Constant.CLIENTS: ['C1', 'C2', 'C3'],                                       Constant.AGGR_CLIENTS: ['C2', 'C3']},
                       {Constant.CLIENTS: ['C1', 'C2', 'C3'],                                                                          }):
            fl_context = self._setup_for_processing_config(config)
            self.controller.process_config(fl_context)
            is_trainer = ( Constant.TRAIN_CLIENTS not in config.keys() ) or ( "C1" in config[Constant.TRAIN_CLIENTS] )
            is_aggregator = (Constant.AGGR_CLIENTS not in config.keys()) or ("C1" in config[Constant.AGGR_CLIENTS])
            self.assertEqual(self.controller.is_trainer, is_trainer)
            self.assertEqual(self.controller.is_aggr, is_aggregator)

    def test_process_config_raises_errors_and_logs(self):
        fl_context = FLContext()
        with self.assertLogs(self.testee_logger, logging.ERROR) as log, self.assertRaises(TypeError) as error:
            self.controller.process_config(fl_context)
        self.assertEqual(log.output[0], "ERROR:swarm_client_ctl:Exception during process_config: argument of type 'NoneType' is not iterable")

    def _setup_for_executing(self, config):
        fl_context = self._setup_for_processing_config(config)
        shareable = Shareable()
        shareable[Constant.CONFIG] = None
        abort_signal = Signal()
        return fl_context, shareable, abort_signal

    def test_execute_returns_if_no_exception(self):
        """
        Test the execute method to ensure proper handling of the task execution flow.
        """
        for task_name, expected_result in (('test_prefix_report_learn_result', {'__headers__': {'__rc__': 'EXECUTION_EXCEPTION'}}),
                                           ('wrong_task_name', {'__headers__': {'__rc__': 'TASK_UNKNOWN'}})):
            config = {Constant.CLIENTS: ['C1', 'C2', 'C3'], Constant.TRAIN_CLIENTS: ['C1', 'C2'], Constant.AGGR_CLIENTS: ['C2', 'C3']}
            fl_context, shareable, abort_signal = self._setup_for_executing(config)
            result = self.controller.execute(task_name, shareable, fl_context, abort_signal)
            self.assertDictEqual(result, expected_result)

    def test_execute_logs_and_returns_on_exception(self):
        config = {Constant.CLIENTS: ['C1', 'C2', 'C3'], Constant.TRAIN_CLIENTS: ['C1', 'C2'], Constant.AGGR_CLIENTS: ['C2', 'C3']}
        fl_context, shareable, abort_signal = self._setup_for_executing(config)

        with self.assertLogs(self.testee_logger, logging.ERROR) as log:
            with mock.patch('swarm_client_ctl.SwarmClientController._process_learn_result', side_effect=Exception('exception')):
                result = self.controller.execute('test_prefix_report_learn_result', shareable, fl_context, abort_signal)
                self.assertEqual(result, make_reply(ReturnCode.EXECUTION_EXCEPTION))
            self.assertEqual(log.output[0], 'ERROR:swarm_client_ctl:Exception during execute: exception')

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
        # TODO cases:
        #      event_type = AppEventType.GLOBAL_BEST_MODEL_AVAILABLE
        #                 ≠
        #      exception logged and raised
        # self.controller.me = "client1"
        # self.fl_ctx.get_prop = MagicMock(return_value="client1")
        # self.fl_ctx.set_prop = MagicMock()
        # self.controller.best_result = None
        # self.controller.handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, self.fl_ctx)
        # self.assertIsNotNone(self.controller.best_result)

    def test_handle_event_logs_and_raises_exception(self):
        fl_context = FLContext()

        with self.assertLogs(self.testee_logger, logging.ERROR) as log, self.assertRaises(Exception) as error:
            with mock.patch('swarm_client_ctl.SwarmClientController.update_status', side_effect=Exception('exception')):
                self.controller.handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, fl_context)
        self.assertEqual(log.output[0], 'ERROR:swarm_client_ctl:Exception during handle_event: exception')

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
