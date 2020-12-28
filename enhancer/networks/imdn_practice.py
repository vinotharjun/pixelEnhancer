from enhancer import *
import enhancer.networks.imdn_blocks as B
from enhancer.networks.esrgan_blocks import Interpolate

class SmallEnhancer(nn.Module):
    def __init__(self,**kwargs):
        super(SmallEnhancer, self).__init__()
        self.in_nc = kwargs.get("in_nc",3)
        self.nf = kwargs.get("nf",64)
        self.num_modules = kwargs.get("num_modules",8)
        self.out_nc = kwargs.get("out_nc",3)
        self.upscale = kwargs.get("upscale",4)
        fea_conv = [B.conv_layer(self.in_nc, self.nf, kernel_size=3)]
        rb_blocks = [B.IMDModule(in_channels=self.nf) for _ in range(self.num_modules)]
        LR_conv = B.conv_layer(self.nf, self.nf, kernel_size=3)
#         upsample_block = B.pixelshuffle_block
#         upsample_block = B.interpolation_block
#         upsampler = upsample_block(self.nf, self.out_nc, upscale_factor=self.upscale)
        if self.upscale == 16:
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
        if self.upscale == 8:
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
        if self.upscale == 2:
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
        if self.upscale == 3:
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
        elif self.upscale == 6:
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
                    ("interpolate1", Interpolate()),
                    ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("interpolate2", Interpolate()),
                    ("up_conv2", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                    ("hrconv_lrelu", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                    ("conv_last", nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True)),
                ]
            )

        upsampler = nn.Sequential(tail_layers)
        self.model = B.sequential(*fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
                                  *upsampler)
        
    def forward(self, inp):
        output = self.model(inp)
        return output
    

