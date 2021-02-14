from enhancer import *


class LossCalculator(nn.Module):
    def __init__(self, **kwargs):
        super(LossCalculator, self).__init__()
        self.loss_array = kwargs.get("loss_details", [])
        if type(self.loss_array) != type([]):
            self.loss_array = [self.loss_array]

    def forward(self, input, target):
        return sum([loss(input, target) for loss in self.loss_array])
