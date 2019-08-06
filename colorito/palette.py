from colorito.utils.io import PaletteReader
from colorito.utils.logging import logger
from colorito.exceptions import GeniePersistException, GenieLoadException

import os
import re
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
        self.color_to_cluster = {}
        self.color_clusters = {}

    def prepare(self):
        self.genie.train(self.palette)

    def update_palette(self, palette):
        self.palette.update(PaletteReader.read_palette(palette))

    def get_color_rgb(self, color):
        """
        Returns the RGB of the color. In case the color is
        not in the palette, approximation is performed.
        :param color:
        :return:
        """
        return self.palette.get(
            color, {
                re.sub(r'\s+', '', col): rgb
                for col, rgb in self.palette
                                    .items()
            }.get(
                re.sub(r'\s+', '', color),
                self.genie
                    .guess_color_rgb(color)
            )
        )

    def get_shades_of(self, color, as_rgb=False):
        """
        Returns similar genies using approximation. If the
        color is part of a color cluster,  returns the co-
        lors in the cluster itself.
        :param color:
        :param as_rgb:
        :return:
        """
        if color in self.genie.color_to_cluster:
            color_cluster = self.genie.color_clusters[
                    self.genie.color_to_cluster[color]
            ]
            return [
                self.get_color_rgb(col) if as_rgb
                else col for col in color_cluster
            ]

        return [
            self.get_color_rgb(shade) if as_rgb else shade
            for shade in self.genie.guess_shades_of(color)
        ]

    def get_main_tint(self, color, as_rgb=False):
        """
        Returns the basic tint closest to the col-
        or. Can be used for color normalization.
        :param color:
        :param as_rgb:
        :return:
        """
        primary_rgbs = self.MAIN_TINTS.values()
        similarities = self.genie.compare(
            self.get_color_rgb(color),
            primary_rgbs
        )

        tint_similarities = zip(
            primary_rgbs if not as_rgb else [*self.MAIN_TINTS],
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
        if color not in self.color_to_cluster:
            self.color_clusters[cluster].add(color)
        else:
            logger.error(
                ' color `{}` already in cluster {} ({})'
                ''.format(
                    color, self.color_to_cluster[color],
                    self.color_clusters[
                        self.color_to_cluster[color]
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
        if self.color_to_cluster.get(color):
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
        Creates a cluster with the specified genies.
        :param colors:
        :param overwrite:
        :return:
        """
        for color in set(colors):
            if overwrite:
                self.mov_color_into_cluster(
                    color,
                    len(self.color_clusters)
                )
            else:
                self.put_color_into_cluster(
                    color,
                    len(self.color_clusters)
                )

    def get_color_cluster_of(self, color):
        """
        Returns the cluster for the color, or None.
        :param color:
        :return:
        """
        return self.color_clusters.get(
             self.color_to_cluster.get(
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
                os.path.join(save_to, self.PALET), 'wb'
            )
        )
        self.genie = genie
