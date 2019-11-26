from colorito import DEFAULT_PALETTE, DEFAULT_NETWORK
from colorito.data.vectorize import NgramVectorizer
from colorito.utils import Reader
from colorito.nnet.model import ColorGenerator
from colorito.colors import Color
from colorito.data.utils import encode

from sklearn.metrics.pairwise import cosine_similarity
from kneed.knee_locator import KneeLocator

import torch
import numpy as np


class SmartPalette(object):

    def __init__(self, colors=DEFAULT_PALETTE, nnet=DEFAULT_NETWORK):
        """
        Initializes a SmartPalette over the provided
        list of colors.  The colors can be specified
        as either a List of Strings, or as a path to
        a text file, where each line contains a colo-
        r name.

        :param colors: list of colors that the palet-
                       te will contain.

        :param nnet: path to neural network weights;
                     leave this unchanged for defaul-
                     t network.
        """

        if isinstance(colors, list):
            self.colors = colors
        if isinstance(colors, str ):
            self.colors = list(
                   Reader.read(
                        colors)
            )

        self.nnet = ColorGenerator.load(nnet)
        n_ = len(self.nnet.encoder.lexicons_)
        self.vectorz = NgramVectorizer(order=n_)
        self.vectorz.lexicons = self.nnet.encoder.lexicons_

        self._index_colors()

    def _index_colors(self):
        """
        Indexes the colors in the palette, mapping
        their names to their hidden representation

        :return:
        """

        col_to_vec = {
            color: vector for color, vector in zip(
                self.colors,
                encode(
                    self.colors,
                    self.vectorz
                )
            )
        }

        X = torch.stack(list(col_to_vec.values()))

        self.index = {
            name: {
                'embed': h.numpy(),
                'color': Color.from_lab(
                    name,
                    l.item(),
                    a.item(),
                    b.item(),
                    unscale=True
                )
            }
            for name, h, (l, a, b) in zip(
                   list(col_to_vec.keys()),
                   self.nnet.h(X),
                   self.nnet( X )
            )
        }

    def search(self, name, **kwargs):
        """
        Searches the palette for colors similar
        to the specified one. The results to be
        returned can be delimited using kwargs:

            1. kwarg `n`: return n-best list;
            2. kwarg `t`: return only colors
                with a similarity greater t-
                han the threshold.

        If no kwarg is specified, the elbow me-
        thod is used to infer how many similar
        colors to be returned.

        :param name:
        :param kwargs:
        :return:
        """
        color_embedding = self.nnet.h(
            encode(name, self.vectorz)
        ).squeeze(0).numpy()

        distances = {
            col: cosine_similarity(
                np.array([color_embedding]),
                np.array([vector['embed']])
            )[0][0]
            for col, vector in
            self.index.items( )
        }

        distances = sorted(
            distances.items( ),
            key=lambda x: x[1],
            reverse=True
        )

        if kwargs.get('t'):
            result = self._threshold(
                 distances, **kwargs)
        elif kwargs.get('n'):
            result = self._n_best(distances, **kwargs)
        else:
            result = self._infer(distances)

        colors = [self.index[col]['color'] for col, _ in result]
        scores = [dist for _, dist in result]

        return colors, scores

    def invent(self, name, **kwargs):
        """
        Generates a color from the given name.

        :param name:
        :param kwargs:
        :return:
        """
        l, a, b = self.nnet.y(encode(
                 name, self.vectorz)).squeeze(0)

        return Color.from_lab(
            name,
            l.item(),
            a.item(),
            b.item(),
            unscale=True
        )

    @staticmethod
    def _n_best(candidates, n=10):
        return [(col, dist) for col, dist in candidates[:n]]

    @staticmethod
    def _threshold(candidates, t=.5):
        n_best = filter(lambda x: x[1] > t, candidates)
        return [(col, dist) for col, dist in n_best]

    @staticmethod
    def _infer(candidates):
        dists = [dista for col, dista in candidates ]
        index = [i for i, _ in enumerate(candidates)]

        knee_pt = KneeLocator(
            index,
            dists,
            curve='convex',
            direction='decreasing'
        ).knee + 1

        return [(col, dist) for col, dist in candidates[:knee_pt]]
