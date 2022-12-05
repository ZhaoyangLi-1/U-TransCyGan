from inspect import ArgSpec
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
from models.generator import MBConv
from models.generator import conv_3x3_bn
from models.generator import Transformer
import torch.nn.functional as F
from models.diff_aug import DiffAugment

# from einops import rearrange
# from einops.layers.torch import Rearrange
# from generator import MBConv
# from generator import conv_3x3_bn
# from generator import Transformer
# import torch.nn.functional as F
# import argparse
# from diff_aug import DiffAugment


def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv3d(inp, oup, 3, stride, 1, bias=False),
        #nn.LayerNorm([oup, image_size[0], image_size[1], image_size[2]]),
        nn.InstanceNorm3d(oup),
        nn.LeakyReLU(0.2, inplace=True)
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm, image_size, downsample, isTrans):
        super().__init__()
        self.dim = dim
        if downsample:
            image_size = (image_size[0]*2, image_size[1]*2, image_size[2]*2)
        if isTrans:
            self.norm = norm([image_size[0]*image_size[1]*image_size[2], dim])
        else:
            self.norm = norm([dim, image_size[0], image_size[1], image_size[2]])
        # self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool3d(3, 2, 1)
            self.proj = nn.Conv3d(inp, oup, 1, 1, 0, bias=False)
            self.proj = nn.Conv3d(inp, oup, 1, 1, 0, bias=False)
            
        if expansion == 1:
                self.conv = nn.Sequential(
                # dw
                    nn.Conv3d(hidden_dim, hidden_dim, 3, stride,
                            1, groups=hidden_dim, bias=False),
                    nn.LayerNorm([hidden_dim, image_size[0], image_size[1], image_size[2]]),
                    #nn.InstanceNorm3d(hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    # pw-linear
                    nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.LayerNorm([oup, image_size[0], image_size[1], image_size[2]])
                    #nn.InstanceNorm3d(oup)
                )
        else:
            self.conv = nn.Sequential(
            # pw
            # down-sample in the first conv
                nn.Conv3d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.LayerNorm([hidden_dim, image_size[0], image_size[1], image_size[2]]),
                #nn.InstanceNorm3d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.LayerNorm([hidden_dim, image_size[0], image_size[1], image_size[2]]),
                #nn.InstanceNorm3d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.LayerNorm([oup, image_size[0], image_size[1], image_size[2]]),
                #nn.InstanceNorm3d(oup),
            )

        self.conv = PreNorm(inp, self.conv, nn.LayerNorm, image_size, self.downsample, False)
        
    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.id, self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # create embedding
        self.relative_bias_table = nn.Embedding((self.id * self.ih * self.iw)*8, 1)

        # [batch_size, num_heads, q_d * q_h * q_w, k_d * k_h * k_w]
        coords = torch.meshgrid((torch.arange(self.id), torch.arange(self.ih), torch.arange(self.iw)))
        # [q_d * q_h * q_w, 3]
        coords = torch.flatten(torch.stack(coords), 1).permute(1, 0)
        # [q_d * q_h * q_w, k_d * k_h * k_w, 3]
        relative_pos = (coords[:, None, :] - coords[None, :, :])
        relative_pos = relative_pos[:, :, 0] * self.ih * self.iw + relative_pos[:, :, 1] * self.iw + relative_pos[:, :, 2]
        # move the whole values in relative_pos to be non-neagtive
        relative_pos = relative_pos - relative_pos.min()
        self.register_buffer("relative_pos", relative_pos)
        
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        B_, N, C = x.shape
        
        # (num_windows*B, N, 3C)
        qkv = self.to_qkv(x)
        # (B, N, 3, num_heads, C // num_heads)
        qkv = rearrange(qkv, 'b n (c h l) -> b n c h l', c=3, h=self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # Decompose to query/key/vector for attention
        # each of q, k, v has dimension of (B_, num_heads, N, C // num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2] # Why not tuple-unpacking?
        q = q * self.scale
        
        # attn becomes (B_, num_heads, N, N) shape
        # N = M^2
        attn = (q @ k.transpose(-2, -1))

        # Use "gather" for more efficiency on GPUs
        attention_bias = self.relative_bias_table(self.relative_pos)[:, :, 0] # [q_d * q_h * q_w, k_d * k_h * k_w]
        attn = attn + attention_bias
        attn = self.attend(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.id, self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool3d(3, 2, 1)
            self.pool2 = nn.MaxPool3d(3, 2, 1)
            self.proj = nn.Conv3d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c id ih iw -> b (id ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm, image_size, False, True),
            Rearrange('b (id ih iw) c -> b c id ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c id ih iw -> b (id ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm, image_size, False, True),
            Rearrange('b (id ih iw) c -> b c id ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class discriminator(nn.Module):
    def __init__(self, args, image_size, in_channels, num_blocks, channels, block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        self.id, self.ih, self.iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        self.channels = channels
        self.args = args
        self.diff_aug = args.diff_aug

        self.s5_shape =  (self.id // 32, self.ih // 32, self.iw // 32)

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (self.id // 2, self.ih // 2, self.iw // 2))
        self.s1 = self._make_layer(
             block[block_types[0]], channels[0], channels[1], num_blocks[1], (self.id // 4, self.ih // 4, self.iw // 4))
        self.s2 = self._make_layer(
             block[block_types[2]], channels[1], channels[2], num_blocks[2], (self.id // 8, self.ih // 8, self.iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (self.id // 16, self.ih // 16, self.iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (self.id // 32, self.ih // 32, self.iw // 32))
        
        self.out = nn.Linear(channels[4]*self.s5_shape[0]*self.s5_shape[1]*self.s5_shape[2], 1)

    def forward(self, x):
        #print(x.min(), x.max())
        x = DiffAugment(x, self.diff_aug)
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = x.view(-1, x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4])
        x = self.out(x)
        return x
        #return self.sigmod(x)


    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        main = nn.Sequential(
            nn.Conv3d(1, 64, 3, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 3, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 368, 3, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(368, 736, 3, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.main = main
        self.linear = nn.Linear(736*6*8*6, 1)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                module.weight.data.normal_(mean=0.0, std=0.0001)
                module.bias.data.zero_()

    def forward(self, input):
        #print("Dis: min:{}, max:{}".format(input.min(), input.max()))
        output = self.main(input)
        output = output.reshape(-1, output.shape[1]*output.shape[2]*output.shape[3]*output.shape[4])
        output = self.linear(output)
        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_dis(args):
    num_blocks = [2, 2, 3, 5, 2]
    channels = [32, 64, 128, 256, 512]
    return discriminator(args, (96, 128, 96), 1, num_blocks, channels)
    #return Discriminator(args)

def define_dis(args, image_size, num_blocks, channels, in_channel):
    return discriminator(image_size, in_channel, num_blocks, channels)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--diff_aug', type=str, default="interpolation,translation,cutout", help='Data Augmentation')
#     args = parser.parse_args()
#     USE_CUDA = torch.cuda.is_available()
#     device = torch.device("cuda:0" if USE_CUDA else "cpu")
#     img = torch.randn(1, 1, 48, 64, 48).to(device)
#     net = generate_dis(args).to(device)
#     out = net(img)
#     A=1

