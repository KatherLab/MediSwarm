import torch
import torch.nn as nn
from monai.networks.nets import ResNet
from monai.networks.nets.resnet import ResNetBlock, ResNetBottleneck

from models.base_model import BasicClassifier, ModelWrapper

class Resnet(nn.Module):
    
    def __init__(self, model_name: str, num_classes: int, norm: str = "batch"):
        super().__init__()
        
        self.model_name = model_name
        valid_model_names = ["resnet10", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet200"]
        assert self.model_name in valid_model_names, f"Model name must be one of {valid_model_names}"

        # Get default resnet parameters
        # >> model_name: (block, layers, shortcut_type, bias_downsample)
        resnet_params = {
            "resnet10": ("basic", [1, 1, 1, 1], "B", False),
            "resnet18": ("basic", [2, 2, 2, 2], "A", True),
            "resnet34": ("basic", [3, 4, 6, 3], "A", True),
            "resnet50": ("bottleneck", [3, 4, 6, 3], "B", False),
            "resnet101": ("bottleneck", [3, 4, 23, 3], "B", False),
            "resnet152": ("bottleneck", [3, 8, 36, 3], "B", False),
            "resnet200": ("bottleneck", [3, 24, 36, 3], "B", False),
        }
        block, layers, shortcut_type, bias_downsample = resnet_params[self.model_name]
        block = ResNetBlock if block == "basic" else ResNetBottleneck
        
        # Get default norm parameters
        norm_mapper = {
            "group": ("group", {"num_groups": 8}),
            "instance": ("instance", {"affine": True}),
            "batch": "batch"
        }
        _norm = norm_mapper[norm]
        
        # Create resnet model
        self.net = ResNet(                         
            block=block,                  
            layers=layers,                
            block_inplanes=[64, 128, 256, 512],  
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_classes,
            bias_downsample=bias_downsample,
            shortcut_type=shortcut_type,
            norm=_norm
        )
        
        return
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

def create_model(model_name: str, num_classes: int = 3, norm: str = "batch") -> nn.Module:
    model = Resnet(model_name=model_name, num_classes=num_classes, norm=norm)
    wrapped_model = ModelWrapper(backbone=model, in_ch=1, num_classes=num_classes)
    return wrapped_model
