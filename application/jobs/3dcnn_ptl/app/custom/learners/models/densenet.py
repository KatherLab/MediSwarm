from .base_model import BasicClassifier
import monai.networks.nets as nets
import torch
import torch.nn.functional as F

class DenseNet121(BasicClassifier):
    """
    DenseNet121 model for classification tasks.

    Attributes:
        model (nn.Module): The DenseNet model from MONAI.
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            spatial_dims: int = 3,
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
        Initializes the DenseNet121 model with the given parameters.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            spatial_dims (int, optional): Number of spatial dimensions. Defaults to 3.
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
        self.model = nets.DenseNet264(spatial_dims=spatial_dims, in_channels=in_ch, out_channels=out_ch)

    def forward(self, x_in: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the DenseNet121 model.

        Args:
            x_in (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pred_hor = self.model(x_in)
        return pred_hor
