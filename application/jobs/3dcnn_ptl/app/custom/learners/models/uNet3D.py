from .base_model import BasicClassifier
import monai.networks.nets as nets
import torch

class UNet3D(BasicClassifier):
    """
    UNet3D model for 3D segmentation tasks.

    Attributes:
        model (nn.Module): The UNet3D model from MONAI.
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            spatial_dims: int = 3,
            channels: tuple = (16, 32, 64, 128, 256),
            strides: tuple = (2, 2, 2, 2),
            num_res_units: int = 2,
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
        Initializes the UNet3D model with the given parameters.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            spatial_dims (int, optional): Number of spatial dimensions. Defaults to 3.
            channels (tuple, optional): Tuple of channel sizes. Defaults to (16, 32, 64, 128, 256).
            strides (tuple, optional): Tuple of stride sizes. Defaults to (2, 2, 2, 2).
            num_res_units (int, optional): Number of residual units. Defaults to 2.
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
        self.model = nets.UNet(
            dimensions=spatial_dims,
            in_channels=in_ch,
            out_channels=out_ch,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units
        )

    def forward(self, x_in: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the UNet3D model.

        Args:
            x_in (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pred_hor = self.model(x_in)
        return pred_hor

    def _generate_predictions(self, source: torch.Tensor) -> torch.Tensor:
        """
        Generates predictions for the given input tensor.

        Args:
            source (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted tensor.
        """
        return self.forward(source)

    def _step(self, batch: dict, batch_idx: int, phase: str, optimizer_idx: int = 0) -> torch.Tensor:
        """
        Performs a step in the training or validation phase.

        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
            phase (str): Current phase ('train' or 'val').
            optimizer_idx (int, optional): Index of the optimizer. Defaults to 0.

        Returns:
            torch.Tensor: Loss value.
        """
        source, target = batch['source'], batch['target']

        if phase == "train":
            pred = self._generate_predictions(source)
        elif phase == "val":
            pred = self._generate_predictions(source)
        else:
            raise ValueError(f"Invalid phase: {phase}")

        target = target.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(pred).float()  # Cast target to float
        loss = self.loss(pred, target)

        logging_dict = {f"{phase}_loss": loss}

        if phase == "val":
            logging_dict["y_true"] = target
            logging_dict["y_pred"] = pred

        logging_dict = {k: v.mean() for k, v in logging_dict.items()}  # Add this line before logging
        self.log_dict(logging_dict, on_step=(phase == "train"), on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0) -> torch.Tensor:
        """
        Performs a step in the validation phase.

        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
            optimizer_idx (int, optional): Index of the optimizer. Defaults to 0.

        Returns:
            torch.Tensor: Loss value.
        """
        return self._step(batch, batch_idx, "val", optimizer_idx)

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0) -> torch.Tensor:
        """
        Performs a step in the training phase.

        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
            optimizer_idx (int, optional): Index of the optimizer. Defaults to 0.

        Returns:
            torch.Tensor: Loss value.
        """
        return self._step(batch, batch_idx, "train", optimizer_idx)
