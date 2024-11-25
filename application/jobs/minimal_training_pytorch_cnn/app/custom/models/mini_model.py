from models.base_model import BasicClassifier
import torch
import torch.nn as nn
import torchinfo
import math

class MiniCNNForTesting(BasicClassifier):
    def __init__(self,
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
                 acc_kwargs: dict = {"task": "binary"},
                 ):
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler,
                         lr_scheduler_kwargs, aucroc_kwargs, acc_kwargs)

        waste_of_memory = 16
        linear_waste_of_memory = int(math.sqrt(waste_of_memory/4))

        self.model = torch.nn.Sequential(
            nn.Conv2d(1, 3, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*4*4, linear_waste_of_memory),                   # temporary tests,
            nn.Linear(linear_waste_of_memory, linear_waste_of_memory),  # this should not be merged to main
            nn.Linear(linear_waste_of_memory, 1)
        )
        print(self.model)
        torchinfo.summary(self.model, input_size=(1, 1, 18, 18))

    def forward(self, x_in: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x_in)
