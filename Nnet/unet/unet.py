from .unet_parts import Encoder, Decoder

import torch.nn as nn

class UNET(nn.Module):
    def __init__(
        self,
        in_channels : int = 10,
        first_out_channels : int = 64, # number of channels for the output feature map
        exit_channels : int = 1, # number of channels for Decoder's last layer
        downhill : int = 4, # number of downsamping/upsampling layers in Encoder/Decoder
        padding : int = 1
    ):
        super(UNET, self).__init__()
        self.encoder = Encoder(in_channels, first_out_channels, padding=padding, downhill=downhill)
        self.decoder = Decoder(first_out_channels*(2**downhill), first_out_channels*(2**(downhill-1)),
                               exit_channels, padding=padding, uphill=downhill)

    def forward(self, x):
        enc_out, routes = self.encoder(x)
        out = self.decoder(enc_out, routes)
        return out