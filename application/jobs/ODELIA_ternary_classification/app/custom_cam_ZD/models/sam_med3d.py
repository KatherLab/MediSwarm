from models import BasicClassifier
import medim
import torch.nn as nn
    
class _SAMMed3D(nn.Module):
    """SAM-Med3D-based backbones."""

    def __init__(self, n_input_channels: int, num_classes: int, spatial_dims: int, **kwargs):
        super().__init__(n_input_channels, num_classes, spatial_dims, **kwargs)
        ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
        sam_med3d_model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
        self.model = sam_med3d_model.image_encoder

    def forward(self, x):
        return self.model(x)

 
 class SAMBrain3D(BasicClassifier):
    """SAM-Brain3D-based classifier using ViT as backbone."""

    def __init__(
            self,
            n_input_channels: int,
            num_classes: int,
            spatial_dims: int,
            slice_fusion_type="transformer",
            optimizer_kwargs={'lr': 1e-6},
            **kwargs
    ):
        super().__init__(n_input_channels, num_classes, spatial_dims, optimizer_kwargs=optimizer_kwargs, **kwargs)
        self.model = _SAMMed3D(out_ch=num_classes, slice_fusion_type=slice_fusion_type)

    def forward(self, x):
        return self.model(x)
