from typing import Dict, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.app_common.tracking.tracker_types import LogWriterName

from nvflare.client.api import log

class ODELIA_Writer:
    """SummaryWriter mimics the usage of Tensorboard's SummaryWriter.

    Users can replace the import of Tensorboard's SummaryWriter with FLARE's SummaryWriter.
    They would then use SummaryWriter the same as before.
    SummaryWriter will send log records to the FLARE system.
    """

    def add_value(self, tag: str, values: dict, global_step: Optional[int] = None, **kwargs):
        """Sends scalars.

        Args:
            tag (str): The parent name for the tags.
            values (dict): Key-value pair storing the tag and corresponding values.
            global_step (optional, int): Global step value.
            **kwargs: Additional arguments to pass to the receiver side.
        """
        log(
            key=tag,
            value=values,
            data_type=AnalyticsDataType.SCALARS,
            global_step=global_step,
            writer=LogWriterName.TORCH_TB,
            **kwargs,
        )

