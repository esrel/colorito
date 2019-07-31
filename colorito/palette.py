from colorito.utils.io import PaletteReader


class SmartPalette(object):

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
        self.palette = {}
        self.genie = genie

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
        return self.palette.get(color,
                                self.genie.guess_color_rgb(color))

    def get_shades_of(self, color):
        """
        Returns similar colors using approximation.
        :param color:
        :return:
        """
        return self.genie.guess_shades_of(color)

    def get_main_tint(self, color):
        """
        Returns the basic tint closest to the col-
        or. Can be used for color normalization.
        :param color:
        :return:
        """
        similarities = self.genie.compare(
            self.get_color_rgb(color),
            [*self.MAIN_TINTS.values()]
        )

        return sorted(similarities)[-1]
