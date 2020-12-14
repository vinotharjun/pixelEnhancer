import enhancer.networks.imdn_blocks as B
from enhancer import *

class SmallEnhancer(nn.Module):
    def __init__(self, **kwargs):
        super(SmallEnhancer, self).__init__()
        self.in_nc = kwargs.get("in_nc",3)
        self.nf = kwargs.get("nf",20)
        self.num_modules = kwargs.get("num_modules",6)
        self.out_nc = kwargs.get("out_nc",3)
        self.upscale = kwargs.get("upscale",4)
        fea_conv = [B.conv_layer(self.in_nc, self.nf, 3),
                                      nn.ReLU(inplace=True),
                                      B.conv_layer(self.nf, self.nf, 3, stride=1,bias=False)]
        rb_blocks = [B.IMDModule_Large(in_channels=self.nf) for _ in range(self.num_modules)]
        LR_conv = B.conv_layer(self.nf, self.nf, 3,bias=False)
        upsample_block = B.pixelshuffle_block
        upsampler = upsample_block(self.nf,self.out_nc,self.upscale)
        upsampler = upsampler(self.nf,self.out_nc,self.upscale)
#         upsampler = B.PixelShuffle_ICNR(self.nf,self.out_nc,scale=self.upscale,blur=False)
        self.model = B.sequential(*fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
                                  *upsampler)
    def forward(self, input):
        output = self.model(input)
        return output