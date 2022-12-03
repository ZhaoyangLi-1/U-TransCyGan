from asyncio import FastChildWatcher
from turtle import forward
from unittest.mock import patch
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from models.diff_aug import DiffAugment

def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv3d(inp, oup, 3, stride, 1, bias=False),
        nn.LayerNorm([oup, image_size[0], image_size[1], image_size[2]]),
        nn.GELU()
    )

def tranConv_3x3_bn(inp, oup, image_size, upsample=False):
    stride = 1 if upsample == False else 2
    if upsample:
        return nn.Sequential(
            nn.ConvTranspose3d(inp, oup, 3, stride, 1, output_padding=1, bias=False),
            nn.LayerNorm([oup, image_size[0], image_size[1], image_size[2]]),
            nn.GELU()
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose3d(inp, oup, 3, stride, 1, bias=False),
            nn.LayerNorm([oup, image_size[0], image_size[1], image_size[2]]),
            nn.GELU()
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
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class TransPreNorm(nn.Module):
    def __init__(self, dim, fn, norm, image_size, upsample, isTrans):
        super().__init__()
        self.dim = dim
        if upsample and not isTrans:
             image_size = (image_size[0]//2, image_size[1]//2, image_size[2]//2)
        else:
            image_size = image_size
        if isTrans:
            self.norm = norm([image_size[0]*image_size[1]*image_size[2], dim])
        else:
            self.norm = norm([dim, image_size[0], image_size[1], image_size[2]])
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
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
                    nn.GELU(),
                    # pw-linear
                    nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.LayerNorm([oup, image_size[0], image_size[1], image_size[2]]),
                )
        else:
            self.conv = nn.Sequential(
            # pw
            # down-sample in the first conv
                nn.Conv3d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.LayerNorm([hidden_dim, image_size[0], image_size[1], image_size[2]]),
                nn.GELU(),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.LayerNorm([hidden_dim, image_size[0], image_size[1], image_size[2]]),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.LayerNorm([oup, image_size[0], image_size[1], image_size[2]]),
            )

        self.conv = PreNorm(inp, self.conv, nn.LayerNorm, image_size, self.downsample, False)
        
    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class TransMBConv(nn.Module):
    def __init__(self, inp, oup, image_size, upsample=False, expansion=4):
        super().__init__()
        self.upsample = upsample
        self.id, self.ih, self.iw = image_size
        stride = 1 #if self.upsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.upsample:
            self.pool = nn.AvgPool3d(1)
            self.proj = nn.ConvTranspose3d(inp, oup, 1, 1, 0, bias=False)
            self.conv = conv_3x3_bn(oup*2, oup, image_size)
            
        if expansion == 1:
                self.deconv = nn.Sequential(
                # dw
                nn.ConvTranspose3d(hidden_dim, hidden_dim, 3, stride,
                            1, groups=hidden_dim, bias=False),
                    nn.LayerNorm([hidden_dim, image_size[0]//2, image_size[1]//2, image_size[2]//2]),
                    nn.GELU(),
                    # pw-linear
                    nn.ConvTranspose3d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.LayerNorm([hidden_dim, image_size[0]//2, image_size[1]//2, image_size[2]//2]),
                )
        else:
            if self.upsample:
                    self.deconv = nn.Sequential(
                    # pw
                    # up-sample in the first conv
                    nn.ConvTranspose3d(inp, hidden_dim, 1, stride, 0, bias=False),
                    nn.LayerNorm([hidden_dim, image_size[0]//2, image_size[1]//2, image_size[2]//2]),
                    nn.GELU(),
                    # dw
                    nn.ConvTranspose3d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                    nn.LayerNorm([hidden_dim, image_size[0]//2, image_size[1]//2, image_size[2]//2]),
                    nn.GELU(),
                    SE(inp, hidden_dim),
                    # pw-linear
                    nn.ConvTranspose3d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.LayerNorm([oup, image_size[0]//2, image_size[1]//2, image_size[2]//2]),
                )
            else:
                self.deconv = nn.Sequential(
                    # pw
                    # up-sample in the first conv
                    nn.ConvTranspose3d(inp, hidden_dim, 1, stride, 0, bias=False),
                    nn.LayerNorm([hidden_dim, image_size[0], image_size[1], image_size[2]]),
                    nn.GELU(),
                    # dw
                    nn.ConvTranspose3d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                    nn.LayerNorm([hidden_dim, image_size[0], image_size[1], image_size[2]]),
                    nn.GELU(),
                    SE(inp, hidden_dim),
                    # pw-linear
                    nn.ConvTranspose3d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.LayerNorm([oup, image_size[0], image_size[1], image_size[2]]),
                )

        self.deconv = TransPreNorm(inp, self.deconv, nn.LayerNorm, image_size, self.upsample, False)
        
    def forward(self, input):
        if self.upsample:
            x, y = input[0], input[1]
            temp1 = self.proj(self.pool(x))
            temp2 = self.deconv(x)
            upout = self.proj(self.pool(x)) + self.deconv(x)

            diffD = self.id - upout.size()[2]
            diffH = self.ih - upout.size()[3]
            diffW = self.iw - upout.size()[4]

            upout = F.pad(upout, [diffD // 2, diffD - diffD // 2,
                                  diffH // 2, diffH - diffH // 2,
                                  diffW // 2, diffW - diffW // 2])

            x = self.pool(torch.cat([upout, y], dim=1))
            x = self.conv(x)
            return x
        else:
            x = input
            return x + self.deconv(x)



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



class TransTransformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, upsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.id, self.ih, self.iw = image_size
        self.upsample = upsample

        if self.upsample:
            self.pool = nn.AvgPool3d(1)
            self.conv = conv_3x3_bn(inp, oup, image_size)
            self.proj = nn.ConvTranspose3d(inp, oup, 1)

        self.attn = Attention(oup, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c id ih iw -> b (id ih iw) c'),
            TransPreNorm(oup, self.attn, nn.LayerNorm, image_size, self.upsample, True),
            Rearrange('b (id ih iw) c -> b c id ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c id ih iw -> b (id ih iw) c'),
            TransPreNorm(oup, self.ff, nn.LayerNorm, image_size, self.upsample, True),
            Rearrange('b (id ih iw) c -> b c id ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, input):
        if self.upsample:
            x, y = input[0], input[1]
            upout = self.proj(x)

            diffD = self.id - upout.size()[2]
            diffH = self.ih - upout.size()[3]
            diffW = self.iw - upout.size()[4]

            upout = F.pad(upout, [diffD // 2, diffD - diffD // 2,
                                  diffH // 2, diffH - diffH // 2,
                                  diffW // 2, diffW - diffW // 2])

            x = self.pool(torch.cat([upout, y], dim=1))
            x = self.conv(x)
        else:
            x = input + self.attn(input)
        x = x + self.ff(x)
        return x


class ConvTrantGe(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, batch_size, diff_aug, block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        self.id, self.ih, self.iw = image_size
        self.channels=channels
        self.batch_size = batch_size
        block = {'C': MBConv, 'T': Transformer}
        upsample_block = {'C': TransMBConv, 'T': TransTransformer}
        self.diff_aug = diff_aug

        self.downs0 = self._make_layer_down(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (self.id // 2, self.ih // 2, self.iw // 2))
        self.downs1 = self._make_layer_down(
             block[block_types[0]], channels[0], channels[1], num_blocks[1], (self.id // 4, self.ih // 4, self.iw // 4))
        self.downs2 = self._make_layer_down(
             block[block_types[1]], channels[1], channels[2], num_blocks[2], (self.id // 8, self.ih // 8, self.iw // 8))
        self.downs3 = self._make_layer_down(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (self.id // 16, self.ih // 16, self.iw // 16))
        self.downs4 = self._make_layer_down(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (self.id // 32, self.ih // 32, self.iw // 32))
    
        
        self.ups4  = self._make_layer_up(
            upsample_block[block_types[3]], channels[4], channels[3], num_blocks[4], (self.id // 16, self.ih // 16, self.iw // 16))
        self.ups3  = self._make_layer_up(
            upsample_block[block_types[2]], channels[3], channels[2], num_blocks[3], (self.id // 8, self.ih // 8, self.iw // 8))
        self.ups2  = self._make_layer_up(
            upsample_block[block_types[1]], channels[2], channels[1], num_blocks[2], (self.id // 4, self.ih // 4, self.iw // 4))
        self.ups1  = self._make_layer_up(
            upsample_block[block_types[0]], channels[1], channels[0], num_blocks[1], (self.id // 2, self.ih // 2, self.iw // 2))
        self.ups0 = self._make_layer_up(
            tranConv_3x3_bn, channels[0], in_channels, num_blocks[0], (self.id , self.ih, self.iw)) 

        # self.ups0  = nn.ConvTranspose3d(channels[0],  in_channels, 3, 1, 1, bias=False)

        self.out = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='trilinear'),
            nn.Tanh()
        )

    def forward(self, x):
        x = DiffAugment(x, self.diff_aug)
        x = self.downs0(x)
        stage1 = self.downs1(x)
        stage2 = self.downs2(stage1)
        stage3 = self.downs3(stage2)
        stage4 = self.downs4(stage3)

        upstage4 = self.ups4((stage4, stage3))
        del stage3, stage4
        upstage3 = self.ups3((upstage4, stage2))
        del upstage4, stage2
        upstage2 = self.ups2((upstage3, stage1))
        del upstage3, stage1
        upstage1 = self.ups1((upstage2, x))
        del upstage2, x
        upstage0 = self.ups0(upstage1)
        del upstage1
        #print(upstage0.min(), upstage0.max())
        #print("Gen: min:{}, max:{}".format(upstage0.min(), upstage0.max()))
        return  self.out(upstage0) #torch.tanh(upstage0)

    def _make_layer_down(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))

        return nn.Sequential(*layers)
    
    def _make_layer_up(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, upsample=True))
            else:
                layers.append(block(oup, oup, image_size))

        return nn.Sequential(*layers)
    


def generate_ConvTrantGe(args):
    num_blocks = [2, 2, 2, 3, 2]
    # num_blocks =[2, 2, 4, 7, 3] # L
    channels = [32, 64, 128, 256, 512]
    #channels = [32, 64, 128, 256, 512]
    #patch_size = [8, 4, 2, 1]      # D
    #channels  = [128, 128, 256, 512, 1026]
    # return ConvTrantGe((96, 128, 96), 1, num_blocks, channels, args.batch_size)
    return ConvTrantGe((96, 128, 96), 1, num_blocks, channels, 1, 'interpolation')


def define_gen(args, image_size, num_blocks, channels, in_channel):
    return ConvTrantGe(image_size, in_channel, num_blocks, channels, args.batch_size, args.diff_aug)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# if __name__ == '__main__':
#     args=None
#     USE_CUDA = torch.cuda.is_available()
#     device = torch.device("cuda:4" if USE_CUDA else "cpu")
#     img = torch.randn(1, 1, 96, 128, 96).to(device)
#     net = generate_ConvTrantGe(args).to(device)
#     out = net(img)
#     A=1