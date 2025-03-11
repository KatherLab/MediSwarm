from models.base_model import BasicClassifier
import torch
import torch.nn as nn


class CNNForTesting(BasicClassifier):
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 1,
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
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler,
                         lr_scheduler_kwargs, aucroc_kwargs, acc_kwargs)

    def forward(self, x_in: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x_in)


class MiniCNNForTesting(CNNForTesting):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(
            nn.Conv2d(1, 3, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*4*4, 1)
        )


class FixedSizeCNNForTesting(CNNForTesting):
    def __init__(self,
                 artificial_model_size: int):
        super().__init__()

        float_size = 2   # 2 or 4, depending on float size on GPU
        heuristic_factor = 1.03  # to compensate for approximate formula
        linear_size = int(math.sqrt(artificial_model_size/float_size)/heuristic_factor)

        self.model = torch.nn.Sequential(
            nn.Conv2d(1, 3, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*4*4, linear_size),        # temporary tests,
            nn.Linear(linear_size, linear_size),  # this should not be merged to main
            nn.Linear(linear_size, 1)
        )
