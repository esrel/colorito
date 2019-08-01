from abc import ABC, abstractmethod

from colorito.text.color_name_vectorizer import ColorNameVectorizer
from colorito.utils.logging import logger
from colorito.exceptions import GenieLoadException, GeniePersistException, GenieDoesntKnowException

import pickle
import os
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


class AbstractColorGenie(ABC):

    def __init__(self, color_name_vectorizer=None):
        self.is_ready, self.palette = False, None
        self.vectorizer = color_name_vectorizer
        self.model = None

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

    def train(self, palette):
        logger.info(
            ' training {}...'
            ''.format(
                self.__class__
            )
        )
        self.palette = palette
        if self.vectorizer:
            self.vectorizer.fit(
                [*self.palette])

        self.train_model()
        self.is_ready = True

        logger.info(
            ' {} is ready!'
            ''.format(
                self.__class__
            )
        )

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
        from the palette (basically returns genies
        that are similar to `color`).
        :param color: String name of the color.
        :return:
        """
        pass

    @abstractmethod
    def compare(self, color, colors):
        """
        Returns an array of similarity measures be-
        tween `color` and each element in `genies`.
        :param color:
        :param colors:
        :return:
        """
        pass


class ColorGenie(AbstractColorGenie):

    VECTORIZER = 'vectorizer.pickle'

    def __init__(self):
        super().__init__(color_name_vectorizer=ColorNameVectorizer)

    @classmethod
    def name(self):
        return 'color_genie'

    @classmethod
    def load(cls, load_from):
        try:
            genie = pickle.load(
                open(
                    os.path.join(
                        load_from,
                        cls.name()
                    ), 'rb'
                )
            )

            genie.vectorizer = pickle.load(
                open(
                    os.path.join(
                        load_from,
                        cls.VECTORIZER
                    ), 'rb'
                )
            )

            return genie

        except Exception as e:
            raise GenieLoadException(str(e))

    def persist(self, save_to):
        vect = self.vectorizer
        try:
            pickle.dump(
                vect,
                open(
                    os.path.join(
                        save_to,
                        self.VECTORIZER
                    ), 'rb'
                )
            )

            self.vectorizer = None

            pickle.dump(
                self,
                open(
                    os.path.join(
                        save_to,
                        self.name()
                    ), 'rb'
                )
            )

        except Exception as e:
            self.vectorizer = vect
            raise GeniePersistException(str(e))

        self.vectorizer = vect

    def train_model(self):
        pass

    def guess_color_rgb(self, color):
        """
        Vectorizes the color name and all other colors'
        names in the knowledge base and selects, as be-
        st candidate, the one having a vector represen-
        tation closest to that of `color`.
        :param color:
        :return:
        """
        color_vector = self.vectorizer.transform(color)
        other_colors = [*self.palette]

        similarities = [
            cosine_similarity(
                color_vector,
                self.vectorizer
                    .transform(other_color)
            )[0]

            for other_color in other_colors
        ]

        similarities = filter(
            lambda x: x > 0.0,
            similarities
        )

        if not similarities:
            raise GenieDoesntKnowException(
                "couldn't infer rgb for {}"
                "".format(color)
            )

        # get rgb of color in the knowledge base
        # having closest name vector representa-
        # tion to `color`.

        return self.palette[
            other_colors[
                np.argmax(similarities)
            ]
        ]

    def guess_shades_of(self, color):
        pass

    def compare(self, color, colors):
        """
        Returns the delta-E between color
        and colors (these have to be pro-
        vided as RGB triples.
        :param color:
        :param colors:
        :return:
        """
        delta_es = []
        labcolor = convert_color(
            sRGBColor(*color),
            LabColor
        )

        for candidate in colors:
            delta_e = delta_e_cie2000(
                labcolor,
                convert_color(
                    sRGBColor(
                        *candidate
                    ),
                    LabColor
                )
            )
            delta_es.append(delta_e)

        return delta_es
