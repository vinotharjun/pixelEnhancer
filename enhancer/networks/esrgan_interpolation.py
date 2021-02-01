from enhancer import *
from collections import OrderedDict
from .esrgan_blocks import *
import enhancer.networks.imdn_blocks as B


class SmallEnhancer(nn.Module):
    def __init__(self, **kwargs):
        super(SmallEnhancer, self).__init__()
        self.in_nc = kwargs.get("in_nc", 3)
        self.nf = kwargs.get("nf", 64)
        self.nb = kwargs.get("num_modules", 23)
        self.out_nc = kwargs.get("out_nc", 3)
        self.upscale = kwargs.get("upscale", 4)
        self.gc = kwargs.get("gc", 32)
        self.head = RRDBNet(
            in_nc=self.in_nc, out_nc=self.out_nc, nf=self.nf, nb=self.nb, gc=self.gc
        )
        if self.upscale == 2:
            self.tail = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "interpolate1",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        (
                            "hrconv_lrelu",
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ),
                        (
                            "conv_last",
                            nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True),
                        ),
                    ]
                )
            )
            
        elif self.upscale == 3:
            self.tail = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "interpolate1",
                            Interpolate(factor=3, mode="nearest", align_corners=False),
                        ),
                        ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        (
                            "hrconv_lrelu",
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ),
                        (
                            "conv_last",
                            nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True),
                        ),
                    ]
                )
            )
            
        elif self.upscale == 6:
            self.tail = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "interpolate1",
                            Interpolate(factor=3, mode="nearest", align_corners=False),
                        ),
                        ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        (
                            "interpolate2",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv2", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        (
                            "hrconv_lrelu",
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ),
                        (
                            "conv_last",
                            nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True),
                        ),
                    ]
                )
            )
        elif self.upscale == 8:
            self.tail = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "interpolate1",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        (
                            "interpolate2",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv2", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        (
                            "interpolate3",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv3", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu3", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        (
                            "hrconv_lrelu",
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ),
                        (
                            "conv_last",
                            nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True),
                        ),
                    ]
                )
            )
        elif self.upscale == 16:
            self.tail = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "interpolate1",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        (
                            "interpolate2",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv2", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        (
                            "interpolate3",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv3", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu3", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        (
                            "interpolate4",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv4", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu4", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        (
                            "hrconv_lrelu",
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ),
                        (
                            "conv_last",
                            nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True),
                        ),
                    ]
                )
            )
        else:
            self.tail = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "interpolate1",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        (
                            "interpolate2",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv2", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        (
                            "hrconv_lrelu",
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ),
                        (
                            "conv_last",
                            nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True),
                        ),
                    ]
                )
            )

    def forward(self, x):
        out = self.head(x)
        out = self.tail(out)
        return out
