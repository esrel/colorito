from colorito.nnet.modules import SmartModule
from colorito import DEVICE

import torch


class Encoder(SmartModule):

    def __init__(self, lexicons, *args):
        self.lexicons_ = lexicons
        super(Encoder, self).__init__(
                      lexicons, *args)

    def output_size(self):
        """
        Forwards a fake batch through the network to
        find out its output size and then returns it.

        :return:
        """
        fake_batch = torch.ones(1, *self.input_dim)
        fake_batch = fake_batch.long().to( DEVICE )
        with torch.no_grad():
            out, _ = self(fake_batch)

        return out.squeeze().size()
