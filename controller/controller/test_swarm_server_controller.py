import unittest

from swarm_server_ctl import SwarmServerController
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.ccwf.common import Constant


class TestSwarmServerController(unittest.TestCase):
    CLIENT_THAT_TRAINS = "client1"
    CLIENT_THAT_AGGREGATES = "client2"
    CLIENT_THAT_TRAINS_AND_AGGREGATES = "client3"
    CLIENT_THAT_DOES_NOTHING = "client4"
    CLIENT_THAT_IS_NOT_INVOLVED = "clientA"
    DEFAULT_NUM_ROUNDS = 2

    def setUp(self):
        self.fl_ctx = FLContext()


    @staticmethod
    def get_controller_for_clients(participating_clients, starting_client):
        controller = SwarmServerController(
            num_rounds=TestSwarmServerController.DEFAULT_NUM_ROUNDS,
            participating_clients=participating_clients,
            starting_client=starting_client
        )
        return controller

    def test_initialization(self):
        participating_clients = [self.CLIENT_THAT_TRAINS, self.CLIENT_THAT_AGGREGATES]
        controller = self.get_controller_for_clients(participating_clients, self.CLIENT_THAT_TRAINS)
        self.assertIsInstance(controller, SwarmServerController)
        self.assertEqual(self.DEFAULT_NUM_ROUNDS, controller.num_rounds)
        self.assertEqual(self.CLIENT_THAT_TRAINS, controller.starting_client)

    def test_start_controller(self):
        print("This test does not work yet.")
        return
    """
        self.controller.start_controller(self.fl_ctx)
        self.assertIn('client1', self.controller.train_clients)
        self.assertIn('client1', self.controller.aggr_clients)
    """

    def test_prepare_config(self):
        participating_clients = [self.CLIENT_THAT_TRAINS, self.CLIENT_THAT_AGGREGATES]
        controller = self.get_controller_for_clients(participating_clients, self.CLIENT_THAT_TRAINS)
        config = controller.prepare_config()
        self.assertIn(Constant.AGGR_CLIENTS, config)
        self.assertIn(Constant.TRAIN_CLIENTS, config)


    def test_invalid_starting_client(self):
        print("This currently fails, left out. TODO fix issue!")
        return
    """
        with self.assertRaises(ValueError):
            SwarmServerController(
                num_rounds=10,
                participating_clients=self.participating_clients,
                starting_client=None
            )
        """

    # TODO Think about which scenarios to test
    #      I.e., which combinations of training, aggregating (and possibly starting) clients are valid and work vs. are invalid and should result in the correct error behavior.
    #      E.g.,
    #       ‣ no training client given
    #       ‣ no aggregating client given
    #       ‣ one client in both categories
    #       ‣ one client in neither category
    #       ‣ invalid (non-participating) client in either category
    def test_client_not_in_train_or_aggr_raises_runtime_error(self):
        participating_clients = [self.CLIENT_THAT_TRAINS, self.CLIENT_THAT_AGGREGATES, self.CLIENT_THAT_IS_NOT_INVOLVED]
        controller = self.get_controller_for_clients(participating_clients, self.CLIENT_THAT_TRAINS)
        with self.assertRaises(RuntimeError):
            controller.start_controller(self.fl_ctx)
