from enhancer.imports.torch_imports import *


class Loss(nn.Module):
    def __init__(self, **kwargs):
        super(Loss, self).__init__()
        self.loss_wgt = kwargs.get("loss_wgts", 1)
        self.loss = nn.L2Loss()

    def forward(self, input, target):
        loss = self.loss(input, target)
        return self.loss_wgt * loss
