from abc import ABC, abstractmethod

from colorito.utils.logging import logger


class ColorGenie(ABC):

    def __init__(self, color_name_vectorizer=None):
        self.is_ready, self.palette = False, None
        self.vectorizer = color_name_vectorizer

    @classmethod
    @abstractmethod
    def name(cls):
        pass

    @classmethod
    @abstractmethod
    def load(cls, load_from):
        pass

    @abstractmethod
    def persist(self, save_to):
        pass

    def ready(self, palette):
        logger.info(
            ' training {}...'
            ''.format(
                self.__class__
            )
        )
        self.palette = palette
        if self.vectorizer:
            self.vectorizer.fit([*self.palette])

        self.train()
        self.is_ready = True

        logger.info(
            ' {} is ready!'
            ''.format(
                self.__class__
            )
        )

    @abstractmethod
    def train(self):
        """
        Trains the genie as a classifier to per-
        form the color operations.
        Implement this, if a data driven soluti-
        on is desired.
        :return:
        """
        pass

    @abstractmethod
    def guess_color_rgb(self, color):
        """
        Returns the rgb of a color name that's not
        in the palette.
        :param color: String name of the color.
        :return:
        """
        pass

    @abstractmethod
    def guess_shades_of(self, color):
        """
        Returns the names of other shades of color
        from the palette (basically returns genies
        that are similar to `color`).
        :param color: String or Tuple<Int, Int, Int>.
        :return:
        """
        pass

    @abstractmethod
    def compare(self, color, colors):
        """
        Returns an array of similarity measures be-
        tween `color` and all the `colors`.
        :param color: String or Tuple<Int, Int, Int>.
        :param colors: List<type(color)>.
        :return:
        """
        pass
