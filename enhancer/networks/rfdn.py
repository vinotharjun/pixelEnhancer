import torch
import torch.nn as nn
from .rfdn_blocks import *
from .esrgan_blocks import *



class SmallEnhancer(nn.Module):
    def __init__(self, **kwargs):
        super(SmallEnhancer, self).__init__()
        self.in_nc = kwargs.get("in_nc",3)
        self.nf = kwargs.get("nf",64)
        self.num_modules = kwargs.get("num_modules",6)
        self.out_nc = kwargs.get("out_nc",3)
        self.upscale = kwargs.get("upscale",4)
        
        self.fea_conv = conv_layer(self.in_nc, self.nf, kernel_size=3)

        self.B1 = RFDB(in_channels=self.nf)
        self.B2 = RFDB(in_channels=self.nf)
        self.B3 = RFDB(in_channels=self.nf)
        self.B4 = RFDB(in_channels=self.nf)
        self.B5 = RFDB(in_channels=self.nf)
        self.B6 = RFDB(in_channels=self.nf)
        self.c = conv_block(self.nf * 6, self.nf, kernel_size=1, act_type="lrelu")

        self.LR_conv = conv_layer(self.nf, self.nf, kernel_size=3)

        if upscale == 16:
            tail_layers = OrderedDict(
                [
                    ("interpolate1", Interpolate()),
                    ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("interpolate2", Interpolate()),
                    ("up_conv2", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("interpolate3", Interpolate()),
                    ("up_conv3", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("lrelu3", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("interpolate4", Interpolate()),
                    ("up_conv4", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("lrelu4", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("hrconv_lrelu", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("conv_last", nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True)),
                ]
            )
        if upscale == 8:
            tail_layers = OrderedDict(
                [
                    ("interpolate1", Interpolate()),
                    ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("interpolate2", Interpolate()),
                    ("up_conv2", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("interpolate3", Interpolate()),
                    ("up_conv3", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("lrelu3", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("hrconv_lrelu", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("conv_last", nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True)),
                ]
            )
        if upscale == 2:
            tail_layers = OrderedDict(
                [
                    ("interpolate1", Interpolate()),
                    ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("hrconv_lrelu", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("conv_last", nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True)),
                ]
            )
        if upscale == 3:
            tail_layers = OrderedDict(
                [
                    ("interpolate1", Interpolate(factor=3)),
                    ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("hrconv_lrelu", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("conv_last", nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True)),
                ]
            )
        elif upscale == 6:
            tail_layers = OrderedDict(
                [
                    ("interpolate1", Interpolate(factor=3)),
                    ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("interpolate2", Interpolate(factor=2)),
                    ("up_conv2", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("hrconv_lrelu", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("conv_last", nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True)),
                ]
            )
        else:
            tail_layers = OrderedDict(
                [
                    ("interpolate1", Interpolate(factor=4)),
                    ("up_conv1", nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True)),
                    ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("interpolate2", Interpolate()),
                    ("up_conv2", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("hrconv_lrelu", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("conv_last", nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)),
                ]
            )

        self.upsampler = nn.Sequential(tail_layers)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B5(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4,out_B5,out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    
