from abc import ABC, abstractmethod
from colorito.text.color_name_vectorizer import ColorNameVectorizer


class AbstractColorGenie(ABC):

    def __init__(self, color_name_vectorizer=None):
        self.is_ready, self.palette = False, None
        self.vectorizer = color_name_vectorizer
        self.model = None

    def train(self, palette):
        self.palette = palette
        if self.vectorizer:
            self.vectorizer.fit(
                [*self.palette])

        self.train_model()
        self.is_ready = True

    @abstractmethod
    def train_model(self):
        """
        Trains a data driven model to perform the
        color operations. Implement this if a da-
        ta driven solution is desired.
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
        from the palette (basically returns colors
        that are similar to `color`).
        :param color: String name of the color.
        :return:
        """
        pass

    @abstractmethod
    def compare(self, color, colors):
        """
        Returns an array of similarity measures be-
        tween `color` and each element in `colors`.
        :param color:
        :param colors:
        :return:
        """
        pass


class ColorGenie(AbstractColorGenie):

    def __init__(self):
        super().__init__(color_name_vectorizer=ColorNameVectorizer)

    def train_model(self):
        pass

    def guess_color_rgb(self, color):
        pass

    def guess_shades_of(self, color):
        pass

    def compare(self, color, colors):
        pass
