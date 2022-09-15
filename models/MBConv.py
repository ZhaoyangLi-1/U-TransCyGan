from functools import partial
from torch import nn
from torch import Tensor
from typing import Optional
from functools import partial
import torch
from einops import rearrange
from einops import repeat
import copy
from typing import Optional, List
import torch.nn.functional as F
from einops import rearrange


def conv_3x3_3D_bn(in_features: int, out_features: int, stride):
    return nn.Sequential(
        nn.Conv3d(in_features, out_features, 3, stride, 1, bias=False),
        nn.BatchNorm3d(out_features),
        nn.GELU()
    )


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        stride: int,
        norm: nn.Module = nn.BatchNorm3d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):

        super().__init__(
            nn.Conv3d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
            norm(out_features),
            act(),
        )


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class MBConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4, downsample: bool = False):
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features * expansion
        self.stride = 1 if downsample==False else 2
        self.Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1, stride=self.stride)
        self.Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3, stride=self.stride)
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        # narrow -> wide
                        self.Conv1X1BnReLU(in_features, 
                                      expanded_features,
                                      act=nn.ReLU6
                                     ),
                        # wide -> wide
                        self.Conv3X3BnReLU(expanded_features, 
                                      expanded_features, 
                                      groups=expanded_features,
                                      act=nn.ReLU6
                                     ),
                        # here you can apply SE
                        SE(expanded_features, expanded_features),
                        # wide -> narrow
                        self.Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                ),
                nn.ReLU(),
            )
        )