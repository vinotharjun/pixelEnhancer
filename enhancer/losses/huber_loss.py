from enhancer.imports.torch_imports import *


class Loss(nn.Module):
    def __init__(self, **kwargs):
        super(Loss, self).__init__()
        self.loss_wgt = kwargs.get("loss_wgts", 1)
        self.delta = kwargs.get("delta", 0.01)
        self.reduce = kwargs.get("delta", False)

    def forward(self, input, target):
        abs_error = torch.abs(input - target)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        losses = (0.5 * torch.pow(quadratic, 2) + self.delta * linear) * self.loss_wgt
        if self.reduce:
            return torch.mean(losses)
        else:
            return losses
