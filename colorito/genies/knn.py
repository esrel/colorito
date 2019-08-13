from colorito.text.vectorizers.sklearn import ColorNameCountVectorizer
from colorito.genies import ColorGenie
from colorito.utils.convert import ColorConverter
from colorito.utils.logging import logger

from colorito.exceptions import (
    GenieLoadException,
    GeniePersistException,
    GenieDoesntKnowException,
    InvalidColorFormatException
)

import pickle
import os
import numpy as np

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


class KNNColorGenie(ColorGenie):

    VECTORIZER = 'vectorizer.pickle'

    def __init__(self, k=10):
        super().__init__(color_name_vectorizer=ColorNameCountVectorizer())
        self.space, self.distances = {}, {}
        self.k = k

    @classmethod
    def name(cls):
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
                    ), 'wb'
                )
            )

            self.vectorizer = None

            pickle.dump(
                self,
                open(
                    os.path.join(
                        save_to,
                        self.name()
                    ), 'wb'
                )
            )

        except Exception as e:
            self.vectorizer = vect
            raise GeniePersistException(str(e))

        self.vectorizer = vect

    def train(self):
        """
        Builds the KNN space with LabColors. Dis-
        tances are computed as delta_cie_2000 be-
        tween colors.
        :return:
        """
        self.space = {
            col: convert_color(
                sRGBColor(*rgb),
                LabColor
            )

            for col, rgb in self.palette
                                .items()
        }

        logger.info(
            ' setting up nearest neighbour:'
            ' computing pairwise distances '
            'between colors in the palette.'
        )

        for color_a in tqdm(self.space):
            for color_b in self.space:
                if color_a == color_b:
                    continue

                # we map each color to a <other_color, dist-
                # ance, other_color_rgb> triple, for all co-
                # lors in the palette then sort by distance.

                self.distances[color_a] = sorted(
                    self.distances.get(
                        color_a,
                        []
                    ) + [(
                        color_b,
                        delta_e_cie2000(
                            self.space[color_a],
                            self.space[color_b]
                        ),
                        self.palette[
                             color_b]
                    )],

                    key=lambda x: x[1],
                    reverse=True
                )

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

        if not list(filter(
            lambda x: x,
            similarities
        )):
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
        try:
            ColorConverter.check_rgb_code(color)
            is_rgb_code = True
        except InvalidColorFormatException as e:
            is_rgb_code = False

        candidates, compare_with = [], []
        for candidate in self.palette:
            candidates.append(candidate)
            if is_rgb_code:
                compare_with.append(
                    self.palette[
                       candidate]
                )
            else:
                compare_with.append(
                          candidate)

        comparing = {
            candid: simil
            for candid, simil in zip(
                candidates,
                self.compare(color,
                             compare_with)
            )
        }

        # return the k closest colors

        return [
            k for k, v in sorted(
                comparing.items(),
                key=lambda x: x[1]
            )[:self.k]
        ]

    def compare(self, color, colors):
        """
        Returns the delta-E between color
        and colors (either provided as na-
        mes or as RGB triples).
        :param color:
        :param colors:
        :return:
        """
        try:
            ColorConverter.check_rgb_code(color)
            memento = color
            inverse_palette = {
                v: k for k, v in self.palette
                                     .items()
            }

            color = inverse_palette.get(color)
            unknown_rgbs = {
                str(col) for col in colors if
                col not in inverse_palette
            }
            if color is None or unknown_rgbs:
                color = self.project(memento)
                logger.warning(
                    ' some of the colors to '
                    'compare are unknown rgb'
                    ' values - returning rgb'
                    ' codes instead of names'
                )
                return [
                    delta_e_cie2000(
                        color,
                        self.project(
                         other_color)
                    )

                    for other_color in colors
                ]

        except InvalidColorFormatException as e:
            pass

        # recover already computed distances if
        # the colors are known (in the palette)

        precomputed = self.distances.get(
                               color, [])
        rgb_to_dist = {
            d[2]: d[1] for d in precomputed
        }
        col_to_dist = {
            d[0]: d[1] for d in precomputed
        }

        color = self.palette.get(
            color,
            self.guess_color_rgb(
                           color)
        )

        # for unknown colors, recompute the di-
        # stance (delta_e_cie2000).

        scores = []
        for col in colors:
            try:
                ColorConverter.check_rgb_code(col)
                score = rgb_to_dist.get(col)
            except InvalidColorFormatException as e:
                try:
                    col = self.palette[col]
                except KeyError:
                    col = self.guess_color_rgb(col)

                score = col_to_dist.get(col)

            scores.append(
                score or
                delta_e_cie2000(
                    self.project(col),
                    self.project(
                           color)
                )
            )

        return scores

    def project(self, color):
        """
        Projects the RGB color into the
        LabColor space (needed for del-
        ta_e_cie2000 computation).
        :param color:
        :return:
        """
        return convert_color(
            sRGBColor(*color),
            LabColor
        )
