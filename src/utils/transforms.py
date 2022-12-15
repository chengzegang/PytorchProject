
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Callable

def get_transform(config, **kwargs) -> Callable:
    return T.Compose([
        T.Resize(config['image_size']),
        T.CenterCrop(config['image_size']),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def get_target_transform(config, **kwargs) -> Callable:
    return T.Compose([
        T.Resize(config['image_size']),
        T.CenterCrop(config['image_size']),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def inverse_normalize(x: torch.Tensor) -> torch.Tensor:
    x = TF.normalize(x,  [-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])
    x = x.clamp(0, 1)
    return x
