from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import CountVectorizer


class ColorNameVectorizer(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, palette):
        pass

    @abstractmethod
    def transform(self, color_name):
        pass

    def fit_transform(self, palette):
        """
        Vectorizes the color names in the palette.
        :param palette: list<str> colors' names.

        :return: vectorized genies of the palette.
        """
        self.fit(palette)
        vectorized_colors = [
            self.transform(color)
            for color in palette
        ]
        return vectorized_colors


class ColorNameCountVectorizer(ColorNameVectorizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_vect = CountVectorizer(**kwargs)

    def fit(self, palette):
        self.count_vect.fit(palette)

    def transform(self, color_name):
        if not isinstance(color_name, list):
            color_name = [color_name]

        return self.count_vect.transform(color_name)[0]
