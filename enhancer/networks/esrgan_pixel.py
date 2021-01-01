from enhancer import *
from collections import OrderedDict
from .esrgan_blocks import *
import enhancer.networks.imdn_blocks as B

class SmallEnhancer(nn.Module):
    def __init__(self, **kwargs):
        super(SmallEnhancer, self).__init__()
        self.in_nc = kwargs.get("in_nc",3)
        self.nf = kwargs.get("nf",64)
        self.nb = kwargs.get("num_modules",23)
        self.out_nc = kwargs.get("out_nc",3)
        self.upscale = kwargs.get("upscale",4)
        self.gc = kwargs.get("gc",32)
        self.head = RRDBNet(in_nc=self.in_nc, out_nc=self.out_nc, nf=self.nf, nb=self.nb, gc=self.gc)
        upsample_block = B.pixelshuffle_block
        self.tail= upsample_block(self.nf,self.out_nc,self.upscale)
        
    def forward(self, x):
        out = self.head(x)
        out = self.tail(out)
        return out
