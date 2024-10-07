from models import ResNet

def select_model():
    resnet_layers = [3, 4, 6, 3]
    model = ResNet(in_ch=1, out_ch=1, spatial_dims=3, layers=resnet_layers)
    return model
