from colorito.colors import Color
from colorito.utils.logs import setup_logger
from colorito.data.utils import clean

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import torch
import os

logger = setup_logger('dataset')


class ColorDataset(Dataset):

    def __init__(
        self,
        colors,
        vectorizer,
        space='lab'
    ):
        self.vectorizer = vectorizer
        self.colorspace = space

        x, y = self._process(colors)

        assert {len(_) for _ in y} == {3}, 'data points\' labels are not colors'
        assert len(x) == len(y), 'different number of data points and of labels'

        x = list(
            pad_sequence(
                self._vectors(x),
                batch_first=True
            )
        )

        assert len({len(_) for _ in x}) == 1, 'data points have different sizes'

        self.x = x
        self.y = y

        self.xlen = len(
              self.x[0])

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

    def _process(self, colors):
        """
        Cleans colors' names, unifies colors
        with the same name (drop duplicates).

        :param colors:
        :return:
        """

        logger.info(' processing colors...')
        logger.info(
            f' got {len(colors)} colors...'
        )
        names = clean([color.name for color in colors])
        labels = [
            self._color_to_space(
                           color) for color in colors
        ]
        colors = zip(
            names,
            labels
        )

        logger.info(' de-duplicating colors...')

        unique = {}
        counts = {}
        for name, val in colors:
            if name not in unique:
                unique[name] = []
                counts[name] = {}

            val_as_string = ", ".join([
                f'{v}' for v in val.tolist()
            ])

            unique[name].append(val)
            counts[name][val_as_string] = counts[name].get(
                                          val_as_string, 0) + 1

        # take as label for duplicated colors, the one that ap-
        # pears with highest frequency, breaking ties randomly:

        unique = {
            k: torch.tensor([
                float(s)
                for s in sorted(
                    counts[k].items(),
                    key=lambda t: t[1]
                )[-1][0].split( ", " )
            ])
            for k, v in unique.items()
        }

        logger.info(
            f' there are {len(unique)} samples'
            f' left after the de-duplication. '
        )

        return (
            list(unique.keys( )),
            list(unique.values())
        )

    def _vectors(self, strings):
        self.vectorizer.fit(strings)
        return self.vectorizer.torch_transform(
                                       strings)

    def _color_to_space(self, color):
        if self.colorspace == 'lab':
            return torch.tensor(color.rescaled_lab)
        if self.colorspace == 'rgb':
            return torch.tensor(color.rescaled_rgb)

        raise ValueError(
            f'Invalid color space'
            f' {self.colorspace} '
            f'- must be either:  '
            f'`lab` or `rgb`'
        )

    @classmethod
    def build(cls, cdir, vectorizer, space='lab'):
        """
        Builds the dataset given a directory con-
        taining csv files with color names and t-
        heir hexadecimal codes.

        :param cdir:
        :param vectorizer:
        :param space:
        :return:
        """

        logger.info(f' gathering data from {cdir}...')

        def read_csv(csvpath):

            with open(csvpath, 'r') as csv:
                lines = csv.read().split('\n')

            read_colors = []

            logger.info(
                f' parsing samples from {csvpath}...'
            )
            for line in filter(lambda x: x, lines[1:]):
                color = Color(*line.split(','))
                read_colors.append(color)

            return read_colors

        colors = []
        for fi in os.listdir(cdir):
            if fi.endswith('.csv'):
                fi = os.path.join(cdir, fi)
                fi_colors = read_csv(fi)
                colors.extend(fi_colors)

        return cls(
            colors,
            vectorizer,
            space=space
        )
