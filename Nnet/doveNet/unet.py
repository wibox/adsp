from torch import nn
from . import unet_scraps as scraps

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, act, first_ch_out=64, alpha=1.0, dropout=True, gradcam=False):
        super(UNet, self).__init__()
        self.first_ch_out = first_ch_out
        self.n_classes = n_classes
        self.act = act
        self.alpha = alpha
        self.gradcam = gradcam
        self.gradients = None
        self.inc = scraps.inconv(n_channels, first_ch_out)
        self.down1 = scraps.down(first_ch_out, first_ch_out * 2, act=act, alpha=alpha)
        self.down2 = scraps.down(first_ch_out * 2, first_ch_out * 4, act=act, alpha=alpha)
        self.down3 = scraps.down(first_ch_out * 4, first_ch_out * 8, act=act, alpha=alpha)
        self.down4 = scraps.down(first_ch_out * 8, first_ch_out * 16, act=act, alpha=alpha)
        self.up1 = scraps.up(first_ch_out * 16, first_ch_out * 8, act=act, alpha=alpha, dropout=dropout)
        self.up2 = scraps.up(first_ch_out * 8, first_ch_out * 4, act=act, alpha=alpha, dropout=dropout)
        self.up3 = scraps.up(first_ch_out * 4, first_ch_out * 2, act=act, alpha=alpha, dropout=dropout)
        self.up4 = scraps.up(first_ch_out * 2, first_ch_out, act=act, alpha=alpha, dropout=dropout)
        self.outc = scraps.outconv(first_ch_out, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.gradcam:
            x.register_hook(self.hook_func)

        x = self.outc(x)
        #return F.sigmoid(x)
        return x

    def hook_func(self, grad):
        self.gradients = grad

    def get_gradients(self):
        return self.gradients

    def get_activation(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x