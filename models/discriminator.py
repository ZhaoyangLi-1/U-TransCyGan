from inspect import ArgSpec
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
from models.generator import MBConv
from models.generator import conv_3x3_bn
from models.generator import Transformer
import torch.nn.functional as F

# from einops import rearrange
# from einops.layers.torch import Rearrange
# from generator import MBConv
# from generator import conv_3x3_bn
# from generator import Transformer
# import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        self.id, self.ih, self.iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (self.id // 2, self.ih // 2, self.iw // 2))
        self.s1 = self._make_layer(
             block[block_types[0]], channels[0], channels[1], num_blocks[1], (self.id // 4, self.ih // 4, self.iw // 4))
        self.s2 = self._make_layer(
             block[block_types[1]], channels[1], channels[2], num_blocks[2], (self.id // 8, self.ih // 8, self.iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (self.id // 16, self.ih // 16, self.iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (self.id // 32, self.ih // 32, self.iw // 32))
        
        #self.s5 = nn.Conv3d(channels[4], 1, kernel_size=(3, 4, 3), padding=1)
        #self.head = nn.Linear(channels[4], 1)
        # self.s5 = nn.Sequential(
        #     nn.Conv3d(channels[4], 64, kernel_size=2, padding=1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        #x = self.s5(x)
        #x = F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)
        return torch.sigmoid(x)
        #return x


    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_dis(args):
    num_blocks = [2, 2, 3, 12, 2]           # L
    channels = [32, 64, 96, 192, 368]       # D
    return CoAtNet((96, 128, 96), 1, num_blocks, channels)


# if __name__ == '__main__':
#     args=None
#     USE_CUDA = torch.cuda.is_available()
#     device = torch.device("cuda:0" if USE_CUDA else "cpu")
#     img = torch.randn(7, 1, 96, 128, 96).to(device)
#     net = generate_dis(args).to(device)
#     out = net(img)
#     A=1

