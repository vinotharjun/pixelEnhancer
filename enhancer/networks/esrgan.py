from enhancer.imports.torch_imports import *
from collections import OrderedDict
from .esrgan_blocks import *


class SuperResolution2x(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(SuperResolution2x, self).__init__()
        self.head = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, gc=gc)
        self.tail = nn.Sequential(
            OrderedDict(
                [
                    ("interpolate1", Interpolate()),
                    ("up_conv1", nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                    ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("hrconv", nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                    ("hrconv_lrelu", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("conv_last", nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)),
                ]
            )
        )

    def forward(self, x):
        x = self.head(x)
        out = self.tail(x)
        return out


class SuperResolution4x(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(SuperResolution4x, self).__init__()
        self.head = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, gc=gc)
        self.tail = nn.Sequential(
            OrderedDict(
                [
                    ("interpolate1", Interpolate()),
                    ("up_conv1", nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                    ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("interpolate2", Interpolate()),
                    ("up_conv2", nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                    ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("hrconv", nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                    ("hrconv_lrelu", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("conv_last", nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)),
                ]
            )
        )

    def forward(self, x):
        x = self.head(x)
        out = self.tail(x)
        return out


class SuperResolution8x(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(SuperResolution8x, self).__init__()
        self.head = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, gc=gc)
        self.tail = nn.Sequential(
            OrderedDict(
                [
                    ("interpolate1", Interpolate()),
                    ("up_conv1", nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                    ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("interpolate2", Interpolate()),
                    ("up_conv2", nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                    ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("interpolate3", Interpolate()),
                    ("up_conv3", nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                    ("lrelu3", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("hrconv", nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                    ("hrconv_lrelu", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("conv_last", nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)),
                ]
            )
        )

    def forward(self, x):
        x = self.head(x)
        out = self.tail(x)
        return out


class SuperResolution16x(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(SuperResolution16x, self).__init__()
        self.head = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, gc=gc)
        self.tail = nn.Sequential(
            OrderedDict(
                [
                    ("interpolate1", Interpolate()),
                    ("up_conv1", nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                    ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("interpolate2", Interpolate()),
                    ("up_conv2", nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                    ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("interpolate3", Interpolate()),
                    ("up_conv3", nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                    ("lrelu3", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("interpolate4", Interpolate()),
                    ("up_conv4", nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                    ("lrelu4", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("hrconv", nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
                    ("hrconv_lrelu", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("conv_last", nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)),
                ]
            )
        )

    def forward(self, x):
        x = self.head(x)
        out = self.tail(x)
        return out