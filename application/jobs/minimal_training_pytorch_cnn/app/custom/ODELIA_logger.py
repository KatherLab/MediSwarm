from typing import Dict, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.app_common.tracking.tracker_types import LogWriterName

#from nvflare.client.api import log

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.utils.analytix_utils import create_analytic_dxo

from typing import Any, Dict, Optional, Tuple

from nvflare.client.api import get_model_registry


def log(key: str, value: Any, data_type: AnalyticsDataType, logger, **kwargs) -> bool:
    """Logs a key value pair.

    We suggest users use the high-level APIs in nvflare/client/tracking.py

    Args:
        key (str): key string.
        value (Any): value to log.
        data_type (AnalyticsDataType): the data type of the "value".
        kwargs: additional arguments to be included.

    Returns:
        whether the key value pair is logged successfully

    Example:

        .. code-block:: python

            log(
                key=tag,
                value=scalar,
                data_type=AnalyticsDataType.SCALAR,
                global_step=global_step,
                writer=LogWriterName.TORCH_TB,
                **kwargs,
            )

    """
    logger.info("ODELIA log: enter")
    model_registry = get_model_registry()
    if model_registry.rank != "0":
        raise RuntimeError("only rank 0 can call log!")

    flare_agent = model_registry.flare_agent
    logger.info("ODELIA log: before create")
    dxo = create_analytic_dxo(tag=key, value=value, data_type=data_type, **kwargs)
    logger.info("ODELIA log: after create")
    return flare_agent.log(dxo)

class ODELIA_Writer:
    """SummaryWriter mimics the usage of Tensorboard's SummaryWriter.

    Users can replace the import of Tensorboard's SummaryWriter with FLARE's SummaryWriter.
    They would then use SummaryWriter the same as before.
    SummaryWriter will send log records to the FLARE system.
    """

    def add_value(self, tag: str, values: dict, logger, global_step: Optional[int] = None, **kwargs):
        """Sends scalars.

        Args:
            tag (str): The parent name for the tags.
            values (dict): Key-value pair storing the tag and corresponding values.
            global_step (optional, int): Global step value.
            **kwargs: Additional arguments to pass to the receiver side.
        """
        logger.info("ODELIA_logger: about to log data")
        log(
            key=tag,
            value=values,
            data_type=AnalyticsDataType.SCALARS,
            global_step=global_step,
            writer=LogWriterName.TORCH_TB,
            logger=logger,
            **kwargs,
        )
        logger.info("ODELIA_logger: data logged")

