from models.base_model import BasicClassifier
import monai.networks.nets as nets
import torch


class ResNet(BasicClassifier):
    """
    ResNet model for classification tasks.

    Attributes:
        model (nn.Module): The ResNet model from MONAI.
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            spatial_dims: int = 3,
            block: str = 'basic',
            layers: list = [3, 4, 6, 3],
            block_inplanes: list = [64, 128, 256, 512],
            feed_forward: bool = True,
            loss=torch.nn.BCEWithLogitsLoss,
            loss_kwargs: dict = {},
            optimizer=torch.optim.AdamW,
            optimizer_kwargs: dict = {'lr': 1e-4},
            lr_scheduler=None,
            lr_scheduler_kwargs: dict = {},
            aucroc_kwargs: dict = {"task": "binary"},
            acc_kwargs: dict = {"task": "binary"}
    ):
        """
        Initializes the ResNet model with the given parameters.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            spatial_dims (int, optional): Number of spatial dimensions. Defaults to 3.
            block (str, optional): Block type for ResNet. Defaults to 'basic'.
            layers (list, optional): List of layer configurations. Defaults to [3, 4, 6, 3].
            block_inplanes (list, optional): List of block in-plane sizes. Defaults to [64, 128, 256, 512].
            feed_forward (bool, optional): Whether to use feed forward. Defaults to True.
            loss (callable, optional): Loss function. Defaults to torch.nn.BCEWithLogitsLoss.
            loss_kwargs (dict, optional): Keyword arguments for the loss function. Defaults to {}.
            optimizer (Optimizer, optional): Optimizer. Defaults to torch.optim.AdamW.
            optimizer_kwargs (dict, optional): Keyword arguments for the optimizer. Defaults to {'lr': 1e-4}.
            lr_scheduler (Scheduler, optional): Learning rate scheduler. Defaults to None.
            lr_scheduler_kwargs (dict, optional): Keyword arguments for the learning rate scheduler. Defaults to {}.
            aucroc_kwargs (dict, optional): Keyword arguments for AUROC. Defaults to {"task": "binary"}.
            acc_kwargs (dict, optional): Keyword arguments for Accuracy. Defaults to {"task": "binary"}.
        """
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler,
                         lr_scheduler_kwargs, aucroc_kwargs, acc_kwargs)
        self.model = nets.ResNet(
            block, layers, block_inplanes, spatial_dims, in_ch, 7, 1, False, 'B', 1.0, out_ch, feed_forward, True
        )

    def forward(self, x_in: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the ResNet model.

        Args:
            x_in (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x_in)
