""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = (DoubleConv(n_channels, 64))
        # self.down1 = (Down(64, 128))
        # self.down2 = (Down(128, 256))
        # self.down3 = (Down(256, 512))
        # factor = 2 if bilinear else 1
        # self.down4 = (Down(512, 1024 // factor))
        # self.up1 = (Up(1024, 512 // factor, bilinear))
        # self.up2 = (Up(512, 256 // factor, bilinear))
        # self.up3 = (Up(256, 128 // factor, bilinear))
        # self.up4 = (Up(128, 64, bilinear))
        # self.outc = (OutConv(64, n_classes))
        scale_factor = 2
        self.inc = (DoubleConv(n_channels, 64//scale_factor))
        self.down1 = (Down(64//scale_factor, 128//scale_factor))
        self.down2 = (Down(128//scale_factor, 256//scale_factor))
        self.down3 = (Down(256//scale_factor, 512//scale_factor))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512//scale_factor, 1024//scale_factor // factor))
        # self.up1 = (Up(1024//scale_factor, 512//scale_factor // factor, bilinear))
        # self.up2 = (Up(512//scale_factor, 256//scale_factor // factor, bilinear))
        # self.up3 = (Up(256//scale_factor, 128//scale_factor // factor, bilinear))
        # self.up4 = (Up(128//scale_factor, 64//scale_factor, bilinear))
        # self.outc = (OutConv(64//scale_factor, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x4_ = self.up1(x5, x4)
        # x3_ = self.up2(x4_, x3)
        # x2_ = self.up3(x3_, x2)
        # x1_ = self.up4(x2_, x1)
        # logits = self.outc(x1_)
        # return logits
        return x1,x2,x3,x4,x5

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        # self.up1 = torch.utils.checkpoint(self.up1)
        # self.up2 = torch.utils.checkpoint(self.up2)
        # self.up3 = torch.utils.checkpoint(self.up3)
        # self.up4 = torch.utils.checkpoint(self.up4)
        # self.outc = torch.utils.checkpoint(self.outc)