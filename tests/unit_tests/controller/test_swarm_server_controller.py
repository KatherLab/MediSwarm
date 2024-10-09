import unittest
import logging
from dataclasses import dataclass

from swarm_server_ctl import SwarmServerController
from nvflare.apis.fl_context import FLContextManager
from nvflare.app_common.ccwf.common import Constant
from nvflare.apis.fl_constant import FLContextKey


@dataclass
class Client:
    name: str


class MockedEngineForTesting:
    def __init__(self, job_id, clients):
        self.job_id = job_id
        self.clients = [Client(i) for i in clients]
        self.fl_ctx_mgr = FLContextManager(engine=self)

    def new_context(self):
        context = self.fl_ctx_mgr.new_context()
        context.set_prop(FLContextKey.WORKFLOW, self.job_id)
        return context

    def get_clients(self):
        return self.clients

    def fire_event(self, _a, _b):
        pass


class TestSwarmServerController(unittest.TestCase):
    CLIENT_THAT_TRAINS                = "client1"
    CLIENT_THAT_AGGREGATES            = "client2"
    CLIENT_THAT_TRAINS_AND_AGGREGATES = "client3"
    CLIENT_THAT_DOES_NOTHING          = "client4"
    CLIENT_THAT_IS_NOT_INVOLVED       = "client5"
    OTHER_CLIENT_THAT_TRAINS          = "client6"
    OTHER_CLIENT_THAT_AGGREGATES      = "client7"
    DEFAULT_NUM_ROUNDS = 2

    def _set_up(self, clients):
        self._engine = MockedEngineForTesting(job_id="UnitTestJob", clients=clients)
        self.fl_ctx = self._engine.new_context()
        self.testee_logger = logging.getLogger("SwarmServerController")

    def setUp(self):
        self._set_up(clients=[])


    def _get_minimum_valid_controller(self):
        participating_clients = [self.CLIENT_THAT_TRAINS, self.CLIENT_THAT_AGGREGATES]
        controller = SwarmServerController(num_rounds=TestSwarmServerController.DEFAULT_NUM_ROUNDS,
                                           participating_clients=participating_clients,
                                           starting_client=self.CLIENT_THAT_TRAINS,
                                           train_clients=[self.CLIENT_THAT_TRAINS],
                                           aggr_clients=[self.CLIENT_THAT_AGGREGATES])
        self._set_up(clients=participating_clients)
        return controller

    def _verify_exception_and_error_log(self, error, log, log_prefix: str, expected_message: str):
        self.assertEqual(expected_message, str(error.exception))
        self.assertIn(f"{log_prefix}: {expected_message}", log.output)

    def _verify_exception_and_error_log_during_constructor(self, error, log, expected_message: str):
        log_prefix = "ERROR:SwarmServerController:Error during initialization"
        self._verify_exception_and_error_log(error, log, log_prefix, expected_message)

    def _initialize_start_finalize(self, controller):
        controller.initialize_run(self.fl_ctx)
        controller.start_controller(self.fl_ctx)
        controller.finalize_run(self.fl_ctx)

    def test_initialization_initializes_correctly(self):
        controller = self._get_minimum_valid_controller()
        self.assertIsInstance(controller, SwarmServerController)
        self.assertEqual(self.DEFAULT_NUM_ROUNDS, controller.num_rounds)
        self.assertEqual(self.CLIENT_THAT_TRAINS, controller.starting_client)

    def test_prepare_config_initializes_correctly(self):
        controller = self._get_minimum_valid_controller()
        config = controller.prepare_config()
        self.assertIn(Constant.AGGR_CLIENTS, config)
        self.assertIn(Constant.TRAIN_CLIENTS, config)

    def test_starting_controller_succeeds(self):
        controller = self._get_minimum_valid_controller()
        controller.initialize_run(self.fl_ctx)
        controller.start_controller(self.fl_ctx)
        self.assertIn(self.CLIENT_THAT_TRAINS, controller.train_clients)
        self.assertIn(self.CLIENT_THAT_AGGREGATES, controller.aggr_clients)
        controller.finalize_run(self.fl_ctx)

    def test_invalid_starting_client_raises_error(self):
        participating_clients = [self.CLIENT_THAT_TRAINS, self.CLIENT_THAT_AGGREGATES]
        controller = SwarmServerController(num_rounds=TestSwarmServerController.DEFAULT_NUM_ROUNDS,
                                           participating_clients=participating_clients,
                                           starting_client=self.CLIENT_THAT_IS_NOT_INVOLVED,
                                           train_clients=[self.CLIENT_THAT_TRAINS],
                                           aggr_clients=[self.CLIENT_THAT_AGGREGATES])
        self._set_up(clients=participating_clients)
        with self.assertRaises(ValueError) as error:
            controller.initialize_run(self.fl_ctx)
        self.assertEqual(f"invalid value '{self.CLIENT_THAT_IS_NOT_INVOLVED}' in 'starting_client'", str(error.exception))
        controller.finalize_run(self.fl_ctx)

    def test_unspecified_training_client_succeeds_initialization(self):
        participating_clients = [self.CLIENT_THAT_AGGREGATES, self.OTHER_CLIENT_THAT_AGGREGATES]
        controller = SwarmServerController(num_rounds=TestSwarmServerController.DEFAULT_NUM_ROUNDS,
                                           participating_clients=participating_clients,
                                           starting_client=self.CLIENT_THAT_AGGREGATES,
                                           # no train_clients given
                                           aggr_clients=participating_clients)
        self._set_up(clients=participating_clients)
        self._initialize_start_finalize(controller)

    def test_no_training_client_succeeds_initialization(self):
        participating_clients = [self.CLIENT_THAT_AGGREGATES, self.OTHER_CLIENT_THAT_AGGREGATES]
        controller = SwarmServerController(num_rounds=TestSwarmServerController.DEFAULT_NUM_ROUNDS,
                                           participating_clients=participating_clients,
                                           starting_client=self.CLIENT_THAT_AGGREGATES,
                                           train_clients=[],
                                           aggr_clients=participating_clients)
        self._set_up(clients=participating_clients)
        self._initialize_start_finalize(controller)

    def test_unspecified_aggregating_client_succeeds_initialization(self):
        participating_clients = [self.CLIENT_THAT_TRAINS, self.OTHER_CLIENT_THAT_TRAINS]
        controller = SwarmServerController(num_rounds=TestSwarmServerController.DEFAULT_NUM_ROUNDS,
                                           participating_clients=participating_clients,
                                           starting_client=self.CLIENT_THAT_TRAINS,
                                           train_clients=participating_clients)  # no aggr_clients given
        self._set_up(clients=participating_clients)
        self._initialize_start_finalize(controller)

    def test_no_aggregating_client_succeeds_initialization(self):
        participating_clients = [self.CLIENT_THAT_TRAINS, self.OTHER_CLIENT_THAT_TRAINS]
        controller = SwarmServerController(num_rounds=TestSwarmServerController.DEFAULT_NUM_ROUNDS,
                                           participating_clients=participating_clients,
                                           starting_client=self.CLIENT_THAT_TRAINS,
                                           train_clients=participating_clients,
                                           aggr_clients=[])
        self._set_up(clients=participating_clients)
        self._initialize_start_finalize(controller)

    def test_uncategorized_client_raises_error(self):
        participating_clients = [self.CLIENT_THAT_TRAINS, self.CLIENT_THAT_AGGREGATES, self.CLIENT_THAT_DOES_NOTHING]
        controller = SwarmServerController(num_rounds=TestSwarmServerController.DEFAULT_NUM_ROUNDS,
                                           participating_clients=participating_clients,
                                           starting_client=self.CLIENT_THAT_TRAINS,
                                           train_clients=[self.CLIENT_THAT_TRAINS],
                                           aggr_clients=[self.CLIENT_THAT_AGGREGATES])
        self._set_up(clients=participating_clients)
        expected_message = f"Config Error: client {self.CLIENT_THAT_DOES_NOTHING} is neither train client nor aggr client"
        log_prefix = f"ERROR:SwarmServerController:[identity=, run=?, wf=UnitTestJob]: Error during start_controller"
        with self.assertLogs(self.testee_logger, logging.DEBUG) as log, self.assertRaises(RuntimeError) as error:
            controller.initialize_run(self.fl_ctx)
        self._verify_exception_and_error_log(error, log, log_prefix, expected_message)
        controller.finalize_run(self.fl_ctx)

    def test_client_can_be_both_training_and_aggregating(self):
        participating_clients = [self.CLIENT_THAT_TRAINS, self.CLIENT_THAT_AGGREGATES, self.CLIENT_THAT_TRAINS_AND_AGGREGATES]
        controller = SwarmServerController(num_rounds=TestSwarmServerController.DEFAULT_NUM_ROUNDS,
                                           participating_clients=participating_clients,
                                           starting_client=self.CLIENT_THAT_TRAINS,
                                           train_clients=[self.CLIENT_THAT_TRAINS, self.CLIENT_THAT_TRAINS_AND_AGGREGATES],
                                           aggr_clients=[self.CLIENT_THAT_AGGREGATES, self.CLIENT_THAT_TRAINS_AND_AGGREGATES])
        self._set_up(clients=participating_clients)
        self._initialize_start_finalize(controller)

    def test_one_participating_client_fails_initialization(self):
        with self.assertLogs(self.testee_logger, logging.DEBUG) as log, self.assertRaises(ValueError) as error:
            _ = SwarmServerController(num_rounds=TestSwarmServerController.DEFAULT_NUM_ROUNDS,
                                      participating_clients=[self.CLIENT_THAT_TRAINS],
                                      starting_client=self.CLIENT_THAT_TRAINS)
        expected_message = f"Not enough participating_clients: must > 1, but got ['{self.CLIENT_THAT_TRAINS}']"
        self._verify_exception_and_error_log_during_constructor(error, log, expected_message)

    def test_error_in_prepare_config_is_raised(self):
        controller = self._get_minimum_valid_controller()
        del controller.train_clients  # do something (that usually would not make any sense) to trigger an exception/error thrown in prepare_config
        expected_message = "'SwarmServerController' object has no attribute 'train_clients'"
        log_prefix = f"ERROR:SwarmServerController:Error during prepare_config"
        with self.assertLogs(self.testee_logger, logging.DEBUG) as log, self.assertRaises(AttributeError) as error:
            controller.prepare_config()
        self._verify_exception_and_error_log(error, log, log_prefix, expected_message)
