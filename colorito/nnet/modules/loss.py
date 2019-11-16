import torch.nn as nn
import torch


class ColorDistance(nn.Module):

    def __init__(self):
        super(ColorDistance, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = 1e-06

    def forward(self, yhat, y):
        # add eps before computing RMSE,
        # to avoid an undefined gradient
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss
