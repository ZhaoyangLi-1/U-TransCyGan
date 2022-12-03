from models.discriminator import generate_dis
from models.generator import generate_ConvTrantGe
import torch
from torch import nn
import itertools

class TransCycleWGANGP(nn.Module):
    def __init__(self, args) -> None:
        self.G_A = generate_ConvTrantGe(args)
        self.G_B = generate_ConvTrantGe(args)
        self.D_A = generate_dis(args)
        self.D_B = generate_dis(args)