import torch.nn as nn
from .swinmonaiedited import SwinUNETR

from models.base_model import BasicClassifier, ModelWrapper

class SwinUNETRMultiTask(nn.Module):
    def __init__(self, img_size, in_channels, out_seg_channels, out_cls_classes, feature_size=48):
        super().__init__()
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_seg_channels,
            spatial_dims=2,
            
        )

        # Classification head
        # Use output of encoder (swinViT)
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Pool over patch tokens
            nn.Flatten(),
            nn.Linear(384, out_cls_classes)  # 768 depends on config
        )
    
    def forward(self, x):
        seg_output, hidden_states = self.swin_unetr(x)  # [B, num_classes, H, W]
        cls_output = self.classifier_head(hidden_states)  # [B, num_classes]
        return cls_output # seg_output, cls_output, hidden_states
    

def create_model(img_size: int, num_classes: int = 3, n_input_channels = 1) -> BasicClassifier:
    model = SwinUNETRMultiTask(img_size=img_size, in_channels=n_input_channels, out_seg_channels=2, out_cls_classes=num_classes)
    wrapped_model = ModelWrapper(backbone=model, in_ch=n_input_channels, num_classes=num_classes)
    return wrapped_model
