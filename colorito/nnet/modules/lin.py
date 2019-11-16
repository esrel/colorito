from colorito import DEVICE
from colorito.nnet.modules import SmartModule

import torch
import torch.nn as nn


class LinearModule(SmartModule):

    def __init__(self, dim, *layers):
        super(LinearModule, self).__init__(
            dim, *layers
        )

        h = 256

        input_dim = tuple(dim)
        input_dim = torch.zeros(input_dim)
        self.input_dim = input_dim.numel()

        def _noop(x):
            return x

        if not layers:
            self.deeper = _noop
            w_in = self.input_dim
        else:
            self.deeper = []
            self.layers = []
            self.relus_ = []
            for layer in layers:

                if not self.layers:
                    prev_w_in = self.input_dim
                else:
                    prev_w_in = self.layers[-1].out_features

                self.relus_.append(nn.ReLU())
                self.layers.append(nn.Linear(
                           prev_w_in, layer))

                self.deeper.append(self.layers[-1])
                self.deeper.append(self.relus_[-1])

            w_in = layers[-1]
            self.layers = [ ]
            self.relus_ = [ ]
            self.deeper = nn.Sequential(
                           *self.deeper)

        self.linear = nn.Sequential(
            nn.Linear(w_in, h),
            nn.BatchNorm1d( h),
            nn.Tanh()
        )

        self.output = nn.Sequential(
            nn.Linear(h, 3),
            nn.Sigmoid(   )
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, self.input_dim)

        x = self.deeper(x)
        h = self.linear(x)
        x = self.output(h)

        return x, h
