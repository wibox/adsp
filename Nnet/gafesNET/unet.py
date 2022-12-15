import torch

from unet_components import *
from encoder import Encoder
from decoder import Decoder

class Unet(torch.nn.Module):
    def __init__(self, encoder : Encoder = None, decoder : Decoder = None):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        pass