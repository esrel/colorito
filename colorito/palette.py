from colorito.utils.io import PaletteReader
from colorito.utils.convert import ColorConverter
from colorito.utils.logging import logger
from colorito.exceptions import GeniePersistException, GenieLoadException, InvalidColorFormatException

import os
import re
import shutil
import pickle

from pickle import UnpicklingError


class SmartPalette(object):

    GENIE = 'genie'
    PALET = 'palet.pickle'

    MAIN_TINTS = {
        'red': (255, 0, 0),
        'pink': (255, 192, 203),
        'orange': (255, 165, 0),
        'yellow': (255, 255, 0),
        'purple': (128, 0, 128),
        'green': (0, 128, 0),
        'blue': (0, 0, 255),
        'brown': (165, 42, 42),
        'white': (255, 255, 255),
        'gray': (128, 128, 128)
    }

    def __init__(self, genie):
        self.genie = genie
        self.palette = {}
        self.color_clusters = {}
        self.col_to_cluster = {}

        self.palette.update(
            self.MAIN_TINTS
        )

    @property
    def inverse_palette(self):
        return {
            rgb: col for col, rgb
            in self.palette.items()
        }

    @property
    def rgb_to_cluster(self):
        return {
            self.get_color_rgb(color):
                cluster for color, cluster
            in self.col_to_cluster.items()
        }

    def prepare(self):
        self.genie.ready(self.palette)

    def update_palette(self, palette):
        # don't overwrite any main tint
        self.palette.update({
            k: v for k, v in
            PaletteReader.read_palette(
                               palette).items()

            if k not in self.MAIN_TINTS
        })

    def get_color_rgb(self, color):
        """
        Returns the RGB of the color. In case color is
        not in the palette, approximation is performed.
        :param color:
        :return:
        """
        try:
            ColorConverter.check_rgb_code(color)
            logger.warning(
                ' `{}` is already an rgb code...'
                ''.format(color)
            )
            return color

        except InvalidColorFormatException as e:
            rgb = self.palette.get(color)
            if rgb is None:
                rgb = {
                    re.sub(r'\s+', '', col): rgb
                    for col, rgb in self.palette
                                        .items()
                }.get(
                    re.sub(r'\s+', '', color)
                )
            if rgb is None:
                logger.warning(
                    ' `{}` not found in palette;'
                    ' will try to infer its rgb.'
                    ''.format(color)
                )
                return self.genie.guess_color_rgb(
                                            color)
            return rgb

    def get_shades_of(self, color, as_rgb=False):
        """
        Returns similar genies using approximation. If the
        color is part of a color cluster,  returns the co-
        lors in the cluster itself.
        :param color:
        :param as_rgb:
        :return:
        """
        from_rules = []
        from_genie = [
            self.get_color_rgb(shade) if as_rgb else shade
            for shade in self.genie.guess_shades_of(color)
        ]

        color_cluster = None

        if color in self.col_to_cluster:
            color_cluster = self.color_clusters[
                self.rgb_to_cluster[color]
            ]

        elif color in self.rgb_to_cluster:
            color_cluster = self.color_clusters[
                self.col_to_cluster[color]
            ]

        if color_cluster:
            from_rules = [
                self.get_color_rgb(col) if as_rgb
                else col for col in color_cluster
            ]

        return (
            from_rules,
            from_genie
        )

    def get_main_tint(self, color, as_rgb=False):
        """
        Returns the basic tint closest to the col-
        or. Can be used for color normalization.
        :param color:
        :param as_rgb:
        :return:
        """
        similarities = self.genie.compare(
            color,
            self.MAIN_TINTS
        )

        tint_similarities = zip(
            [*self.MAIN_TINTS] if as_rgb else
            [*self.MAIN_TINTS.values()],
            similarities
        )

        main_tint = sorted(
            tint_similarities,
            key=lambda x: x[1]
        )[-1][0]

        return main_tint

    def put_color_into_cluster(self, color, cluster):
        """
        Places the color into the cluster. If the co-
        lor is already in another cluster, the opera-
        tion is aborted.
        :param color:
        :param cluster:
        :return:
        """
        if color not in self.col_to_cluster:
            self.color_clusters[cluster].add(color)
            self.col_to_cluster[color] = cluster
        else:
            logger.error(
                ' color `{}` already in cluster {} ({})'
                ''.format(
                    color, self.col_to_cluster[color],
                    self.color_clusters[
                        self.col_to_cluster[color]
                    ]
                )
            )

    def mov_color_into_cluster(self, color, cluster):
        """
        Places the color into the cluster. If the co-
        lor is already in another cluster, it's remo-
        ved from the old cluster.
        :param color:
        :param cluster:
        :return:
        """
        if self.col_to_cluster.get(color):
            logger.info(
                ' removing `{}` from cluster {}...'
                ''.format(color, cluster)
            )
            self.color_clusters[
                cluster].remove(color)

            # if the cluster is left empty, del-
            # ete it.
            if not self.color_clusters[cluster]:
                del self.color_clusters[cluster]

        self.put_color_into_cluster(color, cluster)

    def create_color_cluster(self, colors, overwrite=False):
        """
        Creates a cluster with the specified colors.
        :param colors:
        :param overwrite:
        :return:
        """
        cluster_id = max([*self.color_clusters] or [0])
        for color in set(colors):
            if overwrite:
                self.mov_color_into_cluster(
                    color,
                    cluster_id
                )
            else:
                self.put_color_into_cluster(
                    color,
                    cluster_id
                )

    def get_color_cluster_of(self, color):
        """
        Returns the cluster for the color, or None.
        :param color:
        :return:
        """
        return self.color_clusters.get(
               self.col_to_cluster.get(
                   color, -1
               )
        )

    @classmethod
    def load(cls, genie_cls, load_from):
        if not os.path.isdir(load_from):
            logger.error(
                f" {load_from} is not a valid directory"
            )
            return

        genie_dir = os.path.join(
            load_from,
            cls.GENIE
        )
        palet_pkl = os.path.join(
            load_from,
            cls.PALET
        )
        if not os.path.isdir(genie_dir):
            logger.error(
                f" {genie_dir} not found"
            )
            return
        if not os.path.isfile(palet_pkl):
            logger.error(
                f" {palet_pkl} not found"
            )
            return

        try:
            genie = genie_cls.load(genie_dir)
            try:
                palet = pickle.load(
                    open(palet_pkl, 'rb')
                )
                palet.genie = genie
                return palet

            except UnpicklingError:
                logger.error(
                    f" unpickling of {palet_pkl} failed"
                )
                return

        except GenieLoadException:
            logger.error(
                f" couldn't load genie from {genie_dir}"
            )
            return

    def persist(self, save_to, name='my-palette'):
        """
        Saves this SmartPalette in save_to/name/.
        The genie is kept in save_to/name/genie/.
        :param save_to:
        :param name:
        :return:
        """
        if not os.path.isdir(save_to):
            logger.error(
                f' {save_to} is not a directory'
            )
            return

        # create dir for genie and save it there.

        persist_dir = os.path.join(save_to, name)
        os.mkdir(persist_dir)
        os.mkdir(
            os.path.join(
                persist_dir,
                self.GENIE
            )
        )
        try:
            self.genie.persist(
                os.path.join(
                    persist_dir,
                    self.GENIE
                )
            )
        except GeniePersistException as e:
            shutil.rmtree(persist_dir)
            logger.error(
                f' failed to persist genie: {e}'
            )
            return

        # pickle the palette, without the genie
        # (will be loaded separately), and then
        # reset the genie.

        genie = self.genie
        self.genie = None
        pickle.dump(
            self, open(
                os.path.join(persist_dir, self.PALET), 'wb'
            )
        )
        self.genie = genie
