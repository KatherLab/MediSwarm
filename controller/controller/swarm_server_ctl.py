from nvflare.apis.fl_context import FLContext
from nvflare.app_common.ccwf.common import Constant
from nvflare.app_common.ccwf.server_ctl import ServerSideController
from nvflare.fuel.utils.validation_utils import DefaultValuePolicy, normalize_config_arg, validate_candidates


class SwarmServerController(ServerSideController):
    """
    The SwarmServerController class manages the server side of the swarm learning workflow, a decentralized
    form of federated learning. This controller is responsible for managing the overall job status and ensuring
    the proper execution of the learning workflow across multiple rounds.
    """

    def __init__(
        self,
        num_rounds: int,  # Number of training rounds to be performed across the workflow
        start_round: int = 0,  # Initial round to start training (default is 0)
        task_name_prefix=Constant.TN_PREFIX_SWARM,  # Prefix for naming tasks, default is 'swarm'
        start_task_timeout=Constant.START_TASK_TIMEOUT,  # Timeout for starting a task (in seconds)
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,  # Timeout for configuring a task (in seconds)
        task_check_period: float = Constant.TASK_CHECK_INTERVAL,  # Interval for checking task status (in seconds)
        job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,  # Interval for checking job status (in seconds)
        participating_clients=None,  # List of clients participating in the job
        result_clients=None,  # List of clients to receive the final model
        starting_client=None,  # Client responsible for initiating the workflow
        max_status_report_interval: float = Constant.PER_CLIENT_STATUS_REPORT_TIMEOUT,  # Max interval for client status reporting (in seconds)
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,  # Timeout for overall workflow progress (in seconds)
        private_p2p: bool = True,  # Flag to indicate private peer-to-peer communication
        aggr_clients=None,  # Clients designated for aggregation
        train_clients=None,  # Clients designated for training
    ):
        """
        Initializes the SwarmServerController. This includes setting up the base ServerSideController and handling
        client configurations for training and aggregation.
        """
        try:
            # Normalize and validate result_clients and starting_client inputs
            result_clients = normalize_config_arg(result_clients)
            starting_client = normalize_config_arg(starting_client)

            # Initialize the ServerSideController with validated arguments
            super().__init__(
                num_rounds=num_rounds,
                start_round=start_round,
                task_name_prefix=task_name_prefix,
                start_task_timeout=start_task_timeout,
                configure_task_timeout=configure_task_timeout,
                task_check_period=task_check_period,
                job_status_check_interval=job_status_check_interval,
                participating_clients=participating_clients,
                result_clients=result_clients,
                result_clients_policy=DefaultValuePolicy.ALL,
                starting_client=starting_client,
                starting_client_policy=DefaultValuePolicy.ANY,
                max_status_report_interval=max_status_report_interval,
                progress_timeout=progress_timeout,
                private_p2p=private_p2p,
            )

            # If train_clients or aggr_clients are not provided, initialize them as empty lists
            if not aggr_clients:
                aggr_clients = []

            if not train_clients:
                train_clients = []

            # Assign aggregation and training clients
            self.aggr_clients = aggr_clients
            self.train_clients = train_clients
        except Exception as e:
            self.log_error(None, f"Error during initialization: {e}")
            raise

    def start_controller(self, fl_ctx: FLContext):
        """
        Starts the SwarmServerController, initiating the swarm learning process. This method validates the client
        assignments and ensures that every participating client is designated as either a training or aggregation client.
        """
        try:
            # Call the base class method to start the controller
            super().start_controller(fl_ctx)

            # Validate and assign train_clients based on participating_clients
            self.train_clients = validate_candidates(
                var_name="train_clients",
                candidates=self.train_clients,
                base=self.participating_clients,
                default_policy=DefaultValuePolicy.ALL,
                allow_none=False,
            )

            # Validate and assign aggr_clients based on participating_clients
            self.aggr_clients = validate_candidates(
                var_name="aggr_clients",
                candidates=self.aggr_clients,
                base=self.participating_clients,
                default_policy=DefaultValuePolicy.ALL,
                allow_none=False,
            )

            # Ensure every participating client is in at least one category: training or aggregation
            for c in self.participating_clients:
                if c not in self.train_clients and c not in self.aggr_clients:
                    raise RuntimeError(f"Config Error: client {c} is neither train client nor aggr client")
        except Exception as e:
            self.log_error(fl_ctx, f"Error during start_controller: {e}")
            raise

    def prepare_config(self):
        """
        Prepares and returns the configuration for the current swarm learning round, including the lists of
        aggregation and training clients.
        """
        try:
            return {Constant.AGGR_CLIENTS: self.aggr_clients, Constant.TRAIN_CLIENTS: self.train_clients}
        except Exception as e:
            self.log_error(None, f"Error during prepare_config: {e}")
            raise
