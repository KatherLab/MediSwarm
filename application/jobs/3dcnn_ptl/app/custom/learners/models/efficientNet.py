from .base_model import BasicClassifier
import torch
import torch.nn.functional as F
import timm
import monai.networks.nets as nets

class EfficientNet(BasicClassifier):
    """
    EfficientNet model for 2D classification tasks.

    Attributes:
        model (nn.Module): The EfficientNet model from TIMM.
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            spatial_dims: int = 3,
            model_name: str = 'efficientnet_l2',
            pretrained: bool = False,
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
        Initializes the EfficientNet model with the given parameters.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            spatial_dims (int, optional): Number of spatial dimensions. Defaults to 3.
            model_name (str, optional): Name of the EfficientNet model. Defaults to 'efficientnet_l2'.
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to False.
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
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=in_ch, num_classes=out_ch)

    def forward(self, x_in: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the EfficientNet model.

        Args:
            x_in (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        batch_size, _, num_slices, height, width = x_in.shape
        x_in = x_in.view(batch_size * num_slices, 1, height, width)  # Reshape to [batch_size * num_slices, 1, height, width]

        pred_hor = self.model(x_in)  # Process each slice with EfficientNet

        # Reshape the output back to [batch_size, num_slices, out_ch]
        out_ch = pred_hor.shape[1]
        pred_hor = pred_hor.view(batch_size, num_slices, out_ch)

        # Combine the results from each slice (e.g., by averaging or max-pooling)
        combined_pred = torch.mean(pred_hor, dim=1)

        return combined_pred


class EfficientNet3D(BasicClassifier):
    """
    EfficientNet model for 3D classification tasks.

    Attributes:
        model (nn.Module): The EfficientNet model from MONAI.
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            spatial_dims: int = 3,
            blocks_args_str: list = None,
            width_coefficient: float = 1.0,
            depth_coefficient: float = 1.0,
            dropout_rate: float = 0.2,
            image_size: int = 224,
            norm: tuple = ('batch', {'eps': 0.001, 'momentum': 0.01}),
            drop_connect_rate: float = 0.2,
            depth_divisor: int = 8,
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
        Initializes the EfficientNet3D model with the given parameters.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            spatial_dims (int, optional): Number of spatial dimensions. Defaults to 3.
            blocks_args_str (list, optional): List of block arguments. Defaults to None.
            width_coefficient (float, optional): Width coefficient for EfficientNet. Defaults to 1.0.
            depth_coefficient (float, optional): Depth coefficient for EfficientNet. Defaults to 1.0.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
            image_size (int, optional): Image size. Defaults to 224.
            norm (tuple, optional): Normalization configuration. Defaults to ('batch', {'eps': 0.001, 'momentum': 0.01}).
            drop_connect_rate (float, optional): Drop connect rate. Defaults to 0.2.
            depth_divisor (int, optional): Depth divisor. Defaults to 8.
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
        if blocks_args_str is None:
            blocks_args_str = [
                "r1_k3_s11_e1_i32_o16_se0.25",
                "r2_k3_s22_e6_i16_o24_se0.25",
                "r2_k5_s22_e6_i24_o40_se0.25",
                "r3_k3_s22_e6_i40_o80_se0.25",
                "r3_k5_s11_e6_i80_o112_se0.25",
                "r4_k5_s22_e6_i112_o192_se0.25",
                "r1_k3_s11_e6_i192_o320_se0.25"]
        self.model = nets.EfficientNet(blocks_args_str, spatial_dims, in_ch, out_ch,
                                       width_coefficient, depth_coefficient, dropout_rate,
                                       image_size, norm, drop_connect_rate, depth_divisor)

    def forward(self, x_in: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the EfficientNet3D model.

        Args:
            x_in (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pred_hor = self.model(x_in)
        return pred_hor


class EfficientNet3Db7(BasicClassifier):
    """
    EfficientNetB7 model for 3D classification tasks.

    Attributes:
        model (nn.Module): The EfficientNetB7 model from MONAI.
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            spatial_dims: int = 3,
            blocks_args_str: list = None,
            width_coefficient: float = 1.0,
            depth_coefficient: float = 1.0,
            dropout_rate: float = 0.2,
            image_size: int = 224,
            norm: tuple = ('batch', {'eps': 0.001, 'momentum': 0.01}),
            drop_connect_rate: float = 0.2,
            depth_divisor: int = 8,
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
        Initializes the EfficientNet3Db7 model with the given parameters.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            spatial_dims (int, optional): Number of spatial dimensions. Defaults to 3.
            blocks_args_str (list, optional): List of block arguments. Defaults to None.
            width_coefficient (float, optional): Width coefficient for EfficientNet. Defaults to 1.0.
            depth_coefficient (float, optional): Depth coefficient for EfficientNet. Defaults to 1.0.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
            image_size (int, optional): Image size. Defaults to 224.
            norm (tuple, optional): Normalization configuration. Defaults to ('batch', {'eps': 0.001, 'momentum': 0.01}).
            drop_connect_rate (float, optional): Drop connect rate. Defaults to 0.2.
            depth_divisor (int, optional): Depth divisor. Defaults to 8.
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
        if blocks_args_str is None:
            blocks_args_str = [
                "r1_k3_s11_e1_i32_o32_se0.25",
                "r4_k3_s22_e6_i32_o48_se0.25",
                "r4_k5_s22_e6_i48_o80_se0.25",
                "r4_k3_s22_e6_i80_o160_se0.25",
                "r6_k5_s11_e6_i160_o256_se0.25",
                "r6_k5_s22_e6_i256_o384_se0.25",
                "r3_k3_s11_e6_i384_o640_se0.25",
            ]

        self.model = nets.EfficientNet(blocks_args_str, spatial_dims, in_ch, out_ch,
                                       width_coefficient, depth_coefficient, dropout_rate,
                                       image_size, norm, drop_connect_rate, depth_divisor)

    def forward(self, x_in: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the EfficientNet3Db7 model.

        Args:
            x_in (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pred_hor = self.model(x_in)
        return pred_hor
