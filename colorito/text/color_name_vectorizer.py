from abc import ABC, abstractmethod


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
        :param palette: dict<str, tup<int, int, int>>
                        color name to rgb map.

        :return: vectorized genies of the palette.
        """
        self.fit(palette)
        vectorized_colors = [
            self.transform(color)
            for color in palette
        ]
        return vectorized_colors
