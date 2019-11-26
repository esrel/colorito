from colorito import DEVICE
from colorito.nnet.modules.encoders import Encoder

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import torch
import torch.nn as nn


class LSTMEncoder(Encoder):

    def __init__(
        self,
        input_dim,
        lexicons_,
        max_ngram_order=1,
        ret_sequences=False
    ):

        super(LSTMEncoder, self).__init__(
            lexicons_,
            input_dim,
            max_ngram_order,
            ret_sequences
        )

        self.input_dim = input_dim
        self.lexicons_ = lexicons_
        self.ret_sequences = ret_sequences

        # will cat 1-gram, 2-gram, ..., n-gram
        # embeddings starting from the same of-
        # fset together, and feed them to LSTM.

        self.embedding_len = 0
        self.ngram_embedds = []
        self._init_embedd(max_ngram_order)

        embedding_leng = self.embedding_len
        self.slen = input_dim[ 0 ]  # sequence length
        self.elen = embedding_leng  # elements length

        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=self.elen,
            num_layers=self.num_layers,
            hidden_size=self.hidden_dim
        )

        self.h, self.c = None, None

    @property
    def hidden_dim(self):
        return 512

    @property
    def num_layers(self):
        return 2

    def _init_embedd(self, max_order):
        """
        Builds the embedding layers. One layer
        per ngram feature (up to `max_order`).

        :param max_order:
        :return:
        """
        for order in range(max_order):
            # build one embedding layer for e-
            # ach ngram order, while increasi-
            # ng the embedding dimension as t-
            # he ngram order increases:
            embedding_dim = int(32 * (order+1))
            self.embedding_len += embedding_dim
            self.ngram_embedds.append(
                nn.Embedding(
                    num_embeddings=len(self.lexicons_[order]),
                    padding_idx=self.lexicons_[order][ '#' ] ,
                    embedding_dim=embedding_dim
                ).to(DEVICE)
            )

    def _init_hidden(self, batch_size):
        """
        Initializes the hidden layers and cells of
        the lstm with random numbers within [0, 1].

        :param batch_size:
        :return:
        """
        self.h = torch.zeros((self.num_layers, batch_size, self.hidden_dim))
        self.c = torch.zeros((self.num_layers, batch_size, self.hidden_dim))
        self.h.to(DEVICE)
        self.c.to(DEVICE)

    def batchsort(self, batch):
        """
        Sort sequences in the batch by length and returns
        the index to restore their original order. Needed
        for `pack_padded_sequences`

        :param batch:
        :return:
        """
        lengths = torch.tensor([
            len(list(filter(lambda s: torch.max(s), sequence)))
            for sequence in batch
        ])
        sorted_lens, sorted_ix = lengths.sort(descending=True)

        _, unsorted_ix = sorted_ix.sort()
        sorted_batch = batch[ sorted_ix ]

        return sorted_batch, sorted_lens, unsorted_ix

    def compute_embeddings(self, x):
        """
        Computes the embeddings for all ngram
        features and concatenates them togeth-
        er before they are passed to the LSTM.

        :param x:
        :return:
        """

        shape = x.size()[:-1]

        x = x.long()
        x = torch.cat([
            self.ngram_embedds[i](x[:, :, i:i + 1])
            for i in range(x.size()[-1])
        ], -1)

        return x.view(*shape, self.elen)

    def forward(self, x):
        batch_size = x.size()[0]
        self._init_hidden(
               batch_size)

        # compute embedding for each ngram
        # in the sequence, then concatena-
        # te the embeddings for each elem-
        # ent together.
        x = self.compute_embeddings(x)

        x = x.float()  # cast to float before padding;
        x, sorted_lens, unsorted_ix = self.batchsort(x)

        x = pack_padded_sequence(
          x,  sorted_lens,
          batch_first=True
        )

        x, (self.h, self.c) = self.lstm(x, (self.h, self.c))

        x, _ = pad_packed_sequence(
         x, total_length=self.slen,
         batch_first=True)

        x = torch.index_select(x, 0, unsorted_ix)

        if not self.ret_sequences:
            # return last element of sequences
            # but exclude all padding elements
            real_lengths = torch.index_select(
                  sorted_lens, 0, unsorted_ix)

            column_index = real_lengths - 1

            row_index = torch.arange(len(x))

            x = x[ row_index, column_index ]

        return x, (self.h, self.c)


class LiteEncoder(LSTMEncoder):

    def __init__(
        self,
        input_dim,
        lexicons_,
        ngram_order=1,
        ret_sequences=False
    ):
        super(LiteEncoder, self).__init__(
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
