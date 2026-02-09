import os
from typing import List, Optional

from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets.streaming import AnalyticsReceiver

class ODELIAAnalyticsReceiver(AnalyticsReceiver):
    def __init__(self, tb_folder="tb_events", kwargs: Optional[dict] = None, events: Optional[List[str]] = ("fed.analytix_log_stats",)):
        """Receives analytics data from ODELIA swarm.

        Args:
            tb_folder (str): the folder to store tensorboard files.
            kwargs (Optional[dict], optional): keyword arguments, currently not parsed
            events (optional, List[str]): A list of events to be handled by this receiver.

        """
        super().__init__(events=events)
        #self.log_info(f"Init ODELIA Analytics Receiver with {events}")
        self.writers_table = {}
        self.tb_folder = tb_folder
        self.root_log_dir = None

    def initialize(self, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        run_dir = workspace.get_run_dir(fl_ctx.get_job_id())
        root_log_dir = os.path.join(run_dir, self.tb_folder)
        self.log_info(fl_ctx, f"ODELIA Analytics receiver: log directory {root_log_dir}", fire_event=False)
        os.makedirs(root_log_dir, exist_ok=True)
        self.log_info(fl_ctx, f"ODELIA Analytics receiver: log directory {root_log_dir} created", fire_event=False)
        entries = os.listdir(run_dir)
        self.log_info(fl_ctx, f"ODELIA Analytics receiver: entries in {run_dir}: {entries}", fire_event=False)
        self.root_log_dir = root_log_dir

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        """Saves the received data.

        Specific implementations of AnalyticsReceiver will implement save in their own way.

        Args:
            fl_ctx (FLContext): fl context.
            shareable (Shareable): the received message.
            record_origin (str): the sender of this message / record.
        """
        dxo = from_shareable(shareable)
        analytic_data = AnalyticsData.from_dxo(dxo)
        self.log_info(
            fl_ctx,
            #f"received analytics data: {analytic_data} from {record_origin}",
            f"received analytics data from {record_origin}",
            fire_event=False,
        )
        self.log_info(fl_ctx, f"extracted data: {analytic_data.value}", fire_event=False)

    def finalize(self, fl_ctx: FLContext):
        """Finalizes the receiver.

        Called after EventType.END_RUN.

        Args:
            fl_ctx (FLContext): fl context.
        """
        self.log_debug(
            fl_ctx,
            f"finalizing ODELIAAnalyticsReceiver",
            fire_event=False,
        )
