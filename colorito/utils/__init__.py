from colorito.utils.logs import setup_logger

import os


logger = setup_logger('utils')


class Reader(object):

    @staticmethod
    def read(path, sep=','):
        """
        Reads colors' names from a csv file.
        Colors must be in the first column.

        :param path:
        :param sep:
        :return:
        """

        if not os.path.isfile(path):
            raise ValueError(
                f" Could not read colors from "
                f"{path}: is not a valid file."
            )

        with open(path, 'r') as f:

            logger.info(f' reading colors from {path}...')

            colors = filter(
                lambda fline: fline,
                f.read().split('\n')
            )

            for color in colors:
                color = color.split(sep)[0]
                yield color
