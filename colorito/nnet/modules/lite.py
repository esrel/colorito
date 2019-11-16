from colorito import DEVICE
from colorito.nnet.modules.lstm import RecurrentModule

import torch.nn as nn


class LiteRecurrentModule(RecurrentModule):

    def __init__(
        self,
        input_dim,
        lexicons_,
        ngram_order=1,
        ret_sequences=False
    ):
        super(LiteRecurrentModule, self).__init__(
            input_dim=input_dim,
            lexicons_=lexicons_,
            max_ngram_order=ngram_order,
            ret_sequences=ret_sequences
        )

        self.ngram_order = ngram_order

    @property
    def hidden_dim(self):
        return 256

    @property
    def num_layers(self):
        return 2

    def _init_embedd(self, order):
        """
        Build embedding layers. One layer
        only for ngrams of order `order`.

        :param order:
        :return:
        """
        order = order - 1
        embedding_dim = int(32 * 2 ** (order))
        self.embedding_len += embedding_dim
        self.ngram_embedds = nn.Embedding(
            num_embeddings=len(self.lexicons_[order]),
            padding_idx=self.lexicons_[order][ '#' ] ,
            embedding_dim=embedding_dim
        ).to(DEVICE)

    def compute_embeddings(self, x):
        """
        Computes the embeddings for all ngram
        features and concatenates them togeth-
        er before they are passed to the LSTM.

        :param x:
        :return:
        """
        x = x.long()
        x = self.ngram_embedds(
            x[:, :, self.ngram_order - 1]
        )  # select only n-grams features
        # of interest

        return x
