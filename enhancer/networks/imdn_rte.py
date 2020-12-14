from enhancer import *
import enhancer.networks.imdn_blocks as B

class SmallEnhancer(nn.Module):
    def __init__(self, **kwargs):
        super(SmallEnhancer, self).__init__()
        self.in_nc = kwargs.get("in_nc",3)
        self.nf = kwargs.get("nf",12)
        self.num_modules = kwargs.get("num_modules",5)
        self.out_nc = kwargs.get("out_nc",3)
        self.upscale = kwargs.get("upscale",4)
        fea_conv = [B.conv_layer(self.in_nc, self.nf, kernel_size=3)]
        rb_blocks = [B.IMDModule_speed(in_channels=self.nf) for _ in range(self.num_modules)]
        LR_conv = B.conv_layer(self.nf, self.nf, kernel_size=1)
        upsample_block = B.pixelshuffle_block
        upsampler = upsample_block(self.nf, self.out_nc, upscale_factor=self.upscale)
        
        self.model = B.sequential(*fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
                                  *upsampler)

    def forward(self, input):
        output = self.model(input)
        return output