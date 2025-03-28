import unittest
import logging
from mock import mock

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.common import Constant

from swarm_client_ctl import SwarmClientController


TASK_NAME_PREFIX = "test_prefix"
LEARN_TASK_NAME = "test_learn_task"


class MockedEngineForTesting:
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
        self.testee_logger = logging.getLogger("SwarmClientController")

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
        for argument_empty in ("learn_task_name","persistor_id", "shareable_generator_id", "aggregator_id", "metric_comparator_id"):
            with self.assertLogs(self.testee_logger, logging.ERROR) as log, self.assertRaises(ValueError) as error:
                self.setup_controller(**{argument_empty: ""})
            self.assertTrue(log.output[0].startswith(f"ERROR:SwarmClientController:Error during initialization: {argument_empty} must not be empty"))

        for nonpositive_number, value in (("learn_task_timeout", -1.0), ("min_responses_required", 0), ("wait_time_after_min_resps_received", 0.0), ("learn_task_timeout", 0.0)):
            # no need to distinguish between float and int here
            with self.assertLogs(self.testee_logger, logging.ERROR) as log, self.assertRaises(ValueError) as error:
                self.setup_controller(**{nonpositive_number: value})
            self.assertTrue(log.output[0].startswith(f"ERROR:SwarmClientController:Error during initialization: {nonpositive_number} must > 0, but got {value}"))

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
        for config in ({Constant.CLIENTS: ["C1", "C2", "C3"], Constant.TRAIN_CLIENTS: ["C1", "C2"], Constant.AGGR_CLIENTS: ["C2", "C3"]},
                       {Constant.CLIENTS: ["C1", "C2", "C3"], Constant.TRAIN_CLIENTS: ["C2", "C3"], Constant.AGGR_CLIENTS: ["C1", "C3"]},
                       {Constant.CLIENTS: ["C1", "C2", "C3"], Constant.TRAIN_CLIENTS: ["C1", "C2"], Constant.AGGR_CLIENTS: ["C1", "C3"]},
                       {Constant.CLIENTS: ["C1", "C2", "C3"], Constant.TRAIN_CLIENTS: ["C1", "C2"]                                     },
                       {Constant.CLIENTS: ["C1", "C2", "C3"],                                       Constant.AGGR_CLIENTS: ["C2", "C3"]},
                       {Constant.CLIENTS: ["C1", "C2", "C3"],                                                                          }):
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
        self.assertTrue(log.output[0].startswith("ERROR:SwarmClientController:[identity=, run=?]: Exception during process_config"))

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
        for task_name, expected_result in (("test_prefix_report_learn_result", {"__headers__": {"__rc__": "EXECUTION_EXCEPTION"}}),
                                           ("wrong_task_name", {"__headers__": {"__rc__": "TASK_UNKNOWN"}})):
            config = {Constant.CLIENTS: ["C1", "C2", "C3"], Constant.TRAIN_CLIENTS: ["C1", "C2"], Constant.AGGR_CLIENTS: ["C2", "C3"]}
            fl_context, shareable, abort_signal = self._setup_for_executing(config)
            result = self.controller.execute(task_name, shareable, fl_context, abort_signal)
            self.assertDictEqual(result, expected_result)

    def test_execute_logs_and_returns_on_exception(self):
        config = {Constant.CLIENTS: ["C1", "C2", "C3"], Constant.TRAIN_CLIENTS: ["C1", "C2"], Constant.AGGR_CLIENTS: ["C2", "C3"]}
        fl_context, shareable, abort_signal = self._setup_for_executing(config)

        with self.assertLogs(self.testee_logger, logging.ERROR) as log:
            with mock.patch("swarm_client_ctl.SwarmClientController._process_learn_result", side_effect=Exception("exception")):
                result = self.controller.execute("test_prefix_report_learn_result", shareable, fl_context, abort_signal)
                self.assertEqual(result, make_reply(ReturnCode.EXECUTION_EXCEPTION))
        self.assertTrue(log.output[0].startswith("ERROR:SwarmClientController:[identity=, run=?]: Exception during execute"))

    def test_handle_event_unexpected_event_does_not_fail(self):
        fl_context = FLContext()
        result = self.controller.handle_event(AppEventType.LOCAL_BEST_MODEL_AVAILABLE, fl_context)
        self.assertIsNone(result)

    def test_handle_event_expected_event_logged_correctly(self):
        fl_context = FLContext()
        # TODO Unlike error logs, the SwarmClientController logs this only once via self.log_info, as opposed to self.log_error followed by logger.error.
        #      Unify this when unifying logging.
        with self.assertLogs(logging.getLogger("SwarmClientController"), logging.INFO) as log:
            self.controller.handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, fl_context)
        self.assertEqual(log.output[0], "INFO:SwarmClientController:[identity=, run=?]: Got GLOBAL_BEST_MODEL_AVAILABLE: best metric=None")

    def test_handle_event_other_client_affected_does_not_fail(self):
        fl_context = FLContext()
        fl_context.set_prop(Constant.CLIENT, "C1")
        self.controller.me = "C2"
        self.controller.handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, fl_context)

    def test_handle_event_logs_and_raises_exception(self):
        fl_context = FLContext()

        with self.assertLogs(self.testee_logger, logging.ERROR) as log, self.assertRaises(Exception) as error:
            with mock.patch("swarm_client_ctl.SwarmClientController.update_status", side_effect=Exception("exception")):
                self.controller.handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, fl_context)
        self.assertTrue(log.output[0].startswith("ERROR:SwarmClientController:[identity=, run=?]: Exception during handle_event"))

    """
    The start_run, start_workflow, and learn_task methods are not unit-tested (yet)
    because a minimum setup to run them is not straight-forward.
    They are probably better tested as part of an integration test?
    """
