from enhancer.imports.torch_imports import *


class Loss(nn.Module):
    def __init__(self, **kwargs):
        super(Loss, self).__init__()
        self.loss_wgt = kwargs.get("loss_wgts", 2e-8)

    def forward(self, generated, placeholder=None):
        b, c, h, w = generated.size()
        h_tv = torch.pow(
            (generated[:, :, 1:, :] - generated[:, :, : (h - 1), :]), 2
        ).sum()
        w_tv = torch.pow(
            (generated[:, :, :, 1:] - generated[:, :, :, : (w - 1)]), 2
        ).sum()
        loss = (h_tv + w_tv) / (b * c * h * w)
        return self.loss_wgt * loss
