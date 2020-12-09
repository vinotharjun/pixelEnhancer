import torch
import torch.nn as nn
from .rfdn_blocks import *
from .esrgan_blocks import *

class SmallEnhancer(nn.Module):
    def __init__(self, **kwargs):
        super(SmallEnhancer, self).__init__()
        
        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)
        self.in_nc = kwargs.get("in_nc",3)
        self.nf = kwargs.get("nf",64)
        self.num_modules = kwargs.get("num_modules",4)
        self.out_nc = kwargs.get("out_nc",3)
        self.upscale = kwargs.get("upscale",4)
        
        self.B1 = RFDB(in_channels=self.nf)
        self.B2 = RFDB(in_channels=self.nf)
        self.B3 = RFDB(in_channels=self.nf)
        self.B4 = RFDB(in_channels=self.nf)
        self.c = conv_block(self.nf * 4, self.nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(self.nf, self.nf, kernel_size=3)
        
        upsample_block = pixelshuffle_block
        
        self.upsampler = upsample_block(self.nf, self.in_nc, upscale_factor=self.upscale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        out_lr = self.relu(out_lr)
        output = self.upsampler(out_lr)
        
        return output